/*
 * CaSA DDR4 Command Scheduler v2 — C implementation
 *
 * Resource-first slot-filling scheduler. One command per tCK per DIMM.
 * All DDR4 timing constraints enforced. Full model inference scheduled.
 *
 * Key features:
 *   - Min-heap for in-flight items (O(log n) event processing)
 *   - Indexed ready buckets by (dimm, bank, category)
 *   - Counter-based dependency tracking with fan-in barriers
 *   - Proper projection barriers (ALL popcounts must complete)
 *   - Full-row bus reads (120 columns for 12-neuron rows)
 *   - Activation restore RowCopy before each MAJ3
 *   - Multi-DIMM weight distribution
 *   - Corrected 3-row MAJ3 timing (5 commands)
 *   - Correct tFAW counting (3 ACTs per MAJ3, 2 per RowCopy)
 *   - Configurable activation bits (--act-bits 4/8)
 *   - Per-projection milestone timing
 *
 * Build: gcc -O2 -o casa_sched casa_sched.c -lm
 * Run:   ./casa_sched [--layers N] [--dimms N] [--act-bits N] [--seq N] [--debug N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

/* ===================================================================
 * Model constants (known — BitNet b1.58-2B-4T)
 * =================================================================== */

#define D_MODEL     2560
#define D_FF        6912
#define N_HEADS     32
#define HEAD_DIM    80
#define N_LAYERS    24
#define VOCAB_SIZE  32000
#define ACT_BITS    8
#define WEIGHT_ENC  2
/* N_BITPLANES is runtime-configurable via --act-bits.
 * Each weight row stores BOTH encoding halves (pos + neg). MAJ3 on the full
 * row produces AND(pos_weight, activation) and AND(neg_weight, activation)
 * simultaneously. FPGA splits popcounts by column region.
 * So N_BITPLANES = act_bits (not act_bits * weight_enc). */
#define MAX_BITPLANES 8  /* compile-time max for array sizing */
#define ROW_BITS    65536
#define BITS_PER_NEURON (D_MODEL * WEIGHT_ENC)  /* 5120 */
#define NEURONS_PER_ROW (ROW_BITS / BITS_PER_NEURON)  /* 12 */

/* Full-row data transfer: reading a MAJ3 result or writing activation data
 * requires accessing all useful columns. DDR4 burst = 64 bytes per column. */
#define DATA_PER_ROW_BYTES (NEURONS_PER_ROW * BITS_PER_NEURON / 8)  /* 7680 */
#define BURST_BYTES 64
#define COLS_PER_ROW ((DATA_PER_ROW_BYTES + BURST_BYTES - 1) / BURST_BYTES) /* 120 */

/* ===================================================================
 * Configuration — PARAMETRIC values marked [P]
 * =================================================================== */

typedef struct {
    int slot_id;
    int n_banks;        /* derived: n_bank_groups * banks_per_bg */
    int n_bank_groups;  /* 4 */
    int banks_per_bg;   /* 4 */

    /* Standard timing (tCK) */
    int tRCD, tRAS, tRP, tRC;
    int tRRD_S, tRRD_L;
    int tFAW;
    int tCCD_S, tCCD_L;
    int tBurst;
    int tWR;
    int tREFI, tRFC;

    /* Charge-sharing [P] */
    int t_12_maj3, t_23_maj3;
    int t_12_rc, t_23_rc;

    /* Derived */
    int maj3_time;
    int rc_time;
    int bus_rd_time;    /* Full row: tRCD + COLS*tCCD_L + tRP */
    int bus_wr_time;    /* Full row: tRCD + COLS*tCCD_L + tWR + tRP */
    int bus_data_tck;   /* Data bus occupied: COLS * cas_interval */
    int cols_per_row;
    int cas_interval;   /* tCK between CAS commands: tCCD_L (same BG) or tBurst (BG-parallel) */
} DIMMConfig;

typedef struct {
    int popcount_lat;
    int accumulate_lat;
    int rmsnorm_lat;
    int silu_lat;
    int softmax_lat;
    int rope_lat;
    int attn_dot_lat;
    int attn_sv_lat;
} FPGAConfig;

typedef struct {
    int zero_pool_size;     /* 0 or 16 */
    int weight_copies;      /* 1, 8, or 16 */
    int n_bitplanes;        /* activation bits: 4 or 8 (default 8) */
    int popcount_in_dram;   /* 0 = FPGA (read full row), 1 = in-DRAM (read result only) */
    int t_popcount;         /* [P] in-DRAM popcount latency in tCK (bank busy, no bus) */
    int use_popcnt3;        /* 0 = off, 1 = POPCNT3 accumulation per bank per bitplane */
    int t_popcnt3;          /* [P] tCK per POPCNT3 op (MAJ3+XOR3 = ~243 tCK) */
    int use_lisa;           /* [HYPOTHETICAL] 0 = off, 1 = LISA cross-subarray activation distribution */
    int proj_banks;         /* banks per projection: 16 (default), or fewer with LISA */
    int use_pluto;          /* [HYPOTHETICAL] 0 = off, 1 = pLUTo in-DRAM nonlinear */
    int weight_banks;       /* number of banks to distribute weight rows across. */
    int bg_parallel;        /* 0 = conservative (DIMM-global bus_busy)
                             * 1 = bank-group parallel bus (DDR4 BG parallelism via
                             *     tCCD_S interleaving on data bus). */
    int bit_serial;         /* 0 = neuron-packed (12 neurons per row, current layout).
                             * 1 = bit-serial layout — each row holds weights for ONE
                             *     input dim across all d_out output neurons at columns.
                             *     Enables valid POPCNT3 accumulation (NeurIPS 2024). */
    int bs_K;               /* bit-serial batch size K (inputs accumulated per POPCNT3 tree).
                             * Must fit in one subarray with intermediate rows. */
} Layout;

void dimm_init(DIMMConfig *cfg) {
    cfg->n_banks = cfg->n_bank_groups * cfg->banks_per_bg;
    /* 3-row APA: ACT(weight) + t12 + PRE(violated) + t23 + ACT(zero-ref)
     *           + t12 + PRE(violated) + t23 + ACT(activation) + tRCD + tRP
     * 5 commands total (3 ACT + 2 violated PRE) */
    cfg->maj3_time = 1 + cfg->t_12_maj3 + 1 + cfg->t_23_maj3
                   + 1 + cfg->t_12_maj3 + 1 + cfg->t_23_maj3
                   + 1 + cfg->tRCD + cfg->tRP;
    /* SA-mediated RowCopy (2-row): ACT(src) + t12 + PRE(violated) + t23 + ACT(dst) + tRCD + tRP */
    cfg->rc_time = 1 + cfg->t_12_rc + 1 + cfg->t_23_rc + 1 + cfg->tRCD + cfg->tRP;

    /* Full-row bus access: 120 sequential column accesses via page-mode.
     * Column accesses are tCCD_L apart (same bank = same bank group). */
    cfg->cols_per_row = COLS_PER_ROW;
    cfg->cas_interval = cfg->tCCD_L;
    if (cfg->cas_interval < cfg->tBurst) cfg->cas_interval = cfg->tBurst;
    cfg->bus_data_tck = cfg->cols_per_row * cfg->cas_interval;
    cfg->bus_rd_time = cfg->tRCD + cfg->bus_data_tck + cfg->tRP;
    cfg->bus_wr_time = cfg->tRCD + cfg->bus_data_tck + cfg->tWR + cfg->tRP;
}

DIMMConfig default_dimm(int slot) {
    DIMMConfig c;
    memset(&c, 0, sizeof(c));
    c.slot_id = slot;
    c.n_bank_groups = 4;
    c.banks_per_bg = 4;
    /* DRAM Bender runs DDR4 at 1333 MT/s (tCK = 1.5 ns, CK = 667 MHz).
     * All timing in tCK at 1.5 ns. */
    c.tRCD = 9; c.tRAS = 24; c.tRP = 9; c.tRC = 33;
    c.tRRD_S = 4; c.tRRD_L = 6; c.tFAW = 20;
    c.tCCD_S = 4; c.tCCD_L = 5; c.tBurst = 4; c.tWR = 10;
    c.tREFI = 5200; c.tRFC = 233;
    /* Charge-sharing timing [P] — from RowClone characterization.
     * t_23 threshold: 4.5-6 ns = 3-4 tCK at 1.5 ns/tCK. */
    c.t_12_maj3 = 0; c.t_23_maj3 = 0; /* DIMM 0 measured */
    c.t_12_rc = 10; c.t_23_rc = 2; /* DIMM 0 MultiRowInit measured */
    dimm_init(&c);
    return c;
}

FPGAConfig default_fpga(void) {
    FPGAConfig f;
    memset(&f, 0, sizeof(f));
    f.popcount_lat = 4;
    f.accumulate_lat = 2;
    f.rmsnorm_lat = 50;
    f.silu_lat = 20;
    f.softmax_lat = 100;
    f.rope_lat = 30;
    f.attn_dot_lat = 5;
    f.attn_sv_lat = 5;
    return f;
}

/* ===================================================================
 * Work Item
 * =================================================================== */

#define WORK_MAJ3          0
#define WORK_ROWCOPY       1   /* 1-to-1 row copy (intra-subarray) */
#define WORK_BUS_READ      2
#define WORK_BUS_WRITE     3
#define WORK_FPGA_POP      4
#define WORK_FPGA_NL       5
#define WORK_FPGA_ATTN     6
#define WORK_BARRIER       7
#define WORK_MULTI_ROWCOPY 8   /* 1-to-N broadcast row copy (intra-subarray).
                                * Validated by SiMRA paper's MultiRowInit test
                                * for N up to 31 destinations on commodity DDR4.
                                * Same APA timing as RowCopy, multiple rows
                                * activated as destinations simultaneously. */

#define MRC_FANOUT 31  /* SiMRA MultiRowInit: validated up to 31 destinations */

static const char *work_type_name[] = {
    "MAJ3", "RC", "BUS_RD", "BUS_WR", "FPGA_POP", "FPGA_NL", "FPGA_ATTN", "BARRIER", "MULTI_RC"
};

#define MAX_WORK 20000000

typedef struct {
    int work_type;
    int dimm;           /* -1 for FPGA/barrier */
    int bank;           /* -1 for FPGA/barrier */
    int duration;
    int n_remaining_deps;

    int started;
    int completed;
    int start_cycle;
    int end_cycle;
} WorkItem;

/* ===================================================================
 * DIMM State
 * =================================================================== */

#define MAX_BANKS 16
#define MAX_DIMMS 4

typedef struct {
    DIMMConfig cfg;
    int bank_busy[MAX_BANKS];
    int bus_busy;         /* Legacy: entire data bus (conservative) */
    int bus_busy_bg[4];   /* Per-bank-group bus tracking (DDR4 BG parallelism) */
    int cmd_busy;
    int last_4_acts[8];
    int n_acts;
    int last_act_bg[4];
    int last_ref;
    int n_commands;
} DIMMState;

/* ===================================================================
 * Dynamic Bucket — small dynamic array for ready sets
 * =================================================================== */

typedef struct {
    int *items;
    int n;
    int cap;
} Bucket;

static void bucket_init(Bucket *b) {
    b->items = NULL; b->n = 0; b->cap = 0;
}

static void bucket_add(Bucket *b, int wid) {
    if (b->n >= b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->items = (int *)realloc(b->items, b->cap * sizeof(int));
    }
    b->items[b->n++] = wid;
}

static int bucket_pop_min(Bucket *b) {
    if (b->n == 0) return -1;
    int best = 0;
    for (int i = 1; i < b->n; i++)
        if (b->items[i] < b->items[best]) best = i;
    int wid = b->items[best];
    b->items[best] = b->items[--b->n];
    return wid;
}

static void bucket_free(Bucket *b) {
    free(b->items); b->items = NULL; b->n = 0; b->cap = 0;
}

/* ===================================================================
 * Min-Heap — for in-flight items, keyed by end_cycle
 * =================================================================== */

typedef struct { int end_cycle; int wid; } HeapEntry;

typedef struct {
    HeapEntry *data;
    int n;
    int cap;
} MinHeap;

static void heap_init(MinHeap *h) {
    h->n = 0; h->cap = 1024;
    h->data = (HeapEntry *)malloc(h->cap * sizeof(HeapEntry));
}

static void heap_push(MinHeap *h, int end_cycle, int wid) {
    if (h->n >= h->cap) {
        h->cap *= 2;
        h->data = (HeapEntry *)realloc(h->data, h->cap * sizeof(HeapEntry));
    }
    int i = h->n++;
    h->data[i].end_cycle = end_cycle;
    h->data[i].wid = wid;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->data[p].end_cycle <= h->data[i].end_cycle) break;
        HeapEntry tmp = h->data[i]; h->data[i] = h->data[p]; h->data[p] = tmp;
        i = p;
    }
}

static HeapEntry heap_pop(MinHeap *h) {
    HeapEntry top = h->data[0];
    h->data[0] = h->data[--h->n];
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < h->n && h->data[l].end_cycle < h->data[smallest].end_cycle) smallest = l;
        if (r < h->n && h->data[r].end_cycle < h->data[smallest].end_cycle) smallest = r;
        if (smallest == i) break;
        HeapEntry tmp = h->data[i]; h->data[i] = h->data[smallest]; h->data[smallest] = tmp;
        i = smallest;
    }
    return top;
}

static int heap_peek_cycle(MinHeap *h) {
    return h->n > 0 ? h->data[0].end_cycle : INT_MAX;
}

static void heap_free(MinHeap *h) { free(h->data); }

/* ===================================================================
 * Reverse Dependency Index — dynamic per item
 * =================================================================== */

typedef struct {
    int *deps;
    int n;
    int cap;
} RevDeps;

static RevDeps *rev_deps = NULL;

static void init_rev_deps(int max_items) {
    rev_deps = (RevDeps *)calloc(max_items, sizeof(RevDeps));
}

static void add_rev_dep(int from_wid, int to_wid) {
    RevDeps *r = &rev_deps[from_wid];
    if (r->n >= r->cap) {
        r->cap = r->cap ? r->cap * 2 : 2;
        r->deps = (int *)realloc(r->deps, r->cap * sizeof(int));
    }
    r->deps[r->n++] = to_wid;
}

static void free_rev_deps(int n) {
    for (int i = 0; i < n; i++) free(rev_deps[i].deps);
    free(rev_deps);
}

/* ===================================================================
 * Milestone tracker — records barrier completion times
 * =================================================================== */

#define MAX_MILESTONES 512

static struct {
    const char *name;
    int layer;
    int wid;
} milestones[MAX_MILESTONES];
static int n_milestones = 0;

static void track_milestone(const char *name, int layer, int wid) {
    if (n_milestones < MAX_MILESTONES) {
        milestones[n_milestones].name = name;
        milestones[n_milestones].layer = layer;
        milestones[n_milestones].wid = wid;
        n_milestones++;
    }
}

/* ===================================================================
 * Scheduler State
 * =================================================================== */

typedef struct {
    DIMMState dimms[MAX_DIMMS];
    int n_dimms;

    int fpga_pop_busy;
    int fpga_nl_busy;
    int fpga_attn_busy;

    WorkItem *work;
    int n_work;
    int n_completed;
    int n_ready;

    Bucket ready_bus[MAX_DIMMS][MAX_BANKS];
    Bucket ready_act[MAX_DIMMS][MAX_BANKS];
    Bucket ready_fpga[3];

    MinHeap inflight;

    int cycle;
    int total_nops;

    Layout layout;
    int debug_events;
    int debug_count;
} Scheduler;

/* Forward declarations */
static void add_to_ready_bucket(Scheduler *s, int wid);
static void update_ready(Scheduler *s, int completed_wid);

/* ===================================================================
 * Scheduler init
 * =================================================================== */

static void sched_init(Scheduler *s, int n_dimms, Layout layout) {
    memset(s, 0, sizeof(*s));
    s->n_dimms = n_dimms;
    s->layout = layout;

    for (int i = 0; i < n_dimms; i++) {
        s->dimms[i].cfg = default_dimm(i);

        /* bg_parallel: recompute bus_rd/wr_time using tBurst cadence (100% data bus)
         * instead of tCCD_L cadence (80%). Represents DDR4 bank-group CAS
         * interleaving where different-BG CAS at tCCD_S=4 fills the 1-tCK gaps
         * between same-BG bursts. Net: COLS × (tCCD_L - tBurst) tCK savings per read. */
        if (layout.bg_parallel) {
            DIMMConfig *c = &s->dimms[i].cfg;
            c->cas_interval = c->tBurst;  /* BG-parallel: CAS at tCCD_S≈tBurst cadence */
            c->bus_data_tck = c->cols_per_row * c->cas_interval;
            c->bus_rd_time = c->tRCD + c->bus_data_tck + c->tRP;
            c->bus_wr_time = c->tRCD + c->bus_data_tck + c->tWR + c->tRP;
        }

        memset(s->dimms[i].bank_busy, 0, sizeof(s->dimms[i].bank_busy));
        s->dimms[i].bus_busy = 0;
        s->dimms[i].cmd_busy = 0;
        memset(s->dimms[i].bus_busy_bg, 0, sizeof(s->dimms[i].bus_busy_bg));
        s->dimms[i].n_acts = 0;
        s->dimms[i].last_ref = 0;
        for (int j = 0; j < 4; j++)
            s->dimms[i].last_act_bg[j] = -100;
    }

    s->work = (WorkItem *)calloc(MAX_WORK, sizeof(WorkItem));
    s->n_work = 0;
    s->n_completed = 0;
    s->n_ready = 0;

    for (int d = 0; d < MAX_DIMMS; d++)
        for (int b = 0; b < MAX_BANKS; b++) {
            bucket_init(&s->ready_bus[d][b]);
            bucket_init(&s->ready_act[d][b]);
        }
    for (int i = 0; i < 3; i++) bucket_init(&s->ready_fpga[i]);

    heap_init(&s->inflight);
    init_rev_deps(MAX_WORK);
}

static void sched_free(Scheduler *s) {
    for (int d = 0; d < MAX_DIMMS; d++)
        for (int b = 0; b < MAX_BANKS; b++) {
            bucket_free(&s->ready_bus[d][b]);
            bucket_free(&s->ready_act[d][b]);
        }
    for (int i = 0; i < 3; i++) bucket_free(&s->ready_fpga[i]);
    heap_free(&s->inflight);
    free_rev_deps(s->n_work);
    free(s->work);
}

/* ===================================================================
 * Work graph construction
 * =================================================================== */

static int add_work(Scheduler *s, int type, int dimm, int bank, int duration) {
    int wid = s->n_work++;
    if (wid >= MAX_WORK) {
        fprintf(stderr, "FATAL: work item overflow at %d (max %d)\n", wid, MAX_WORK);
        exit(1);
    }
    WorkItem *w = &s->work[wid];
    w->work_type = type;
    w->dimm = dimm;
    w->bank = bank;
    w->duration = duration;
    w->n_remaining_deps = 0;
    w->started = 0;
    w->completed = 0;
    w->start_cycle = -1;
    w->end_cycle = -1;
    return wid;
}

static void add_dep(Scheduler *s, int dep_wid, int item_wid) {
    if (dep_wid < 0) return;
    s->work[item_wid].n_remaining_deps++;
    add_rev_dep(dep_wid, item_wid);
}

static void finalize_ready(Scheduler *s) {
    for (int i = 0; i < s->n_work; i++) {
        if (s->work[i].n_remaining_deps == 0) {
            add_to_ready_bucket(s, i);
        }
    }
}

/* ===================================================================
 * Ready bucket management
 * =================================================================== */

static void add_to_ready_bucket(Scheduler *s, int wid) {
    WorkItem *w = &s->work[wid];

    if (w->work_type == WORK_BARRIER) {
        w->started = 1;
        w->completed = 1;
        w->start_cycle = s->cycle;
        w->end_cycle = s->cycle;
        s->n_completed++;
        if (s->debug_events > 0 && s->debug_count < s->debug_events) {
            printf("  [%d] BARRIER wid=%d completed instantly\n", s->cycle, wid);
            s->debug_count++;
        }
        update_ready(s, wid);
        return;
    }

    s->n_ready++;

    switch (w->work_type) {
        case WORK_BUS_READ:
        case WORK_BUS_WRITE:
            bucket_add(&s->ready_bus[w->dimm][w->bank], wid);
            break;
        case WORK_MAJ3:
        case WORK_ROWCOPY:
        case WORK_MULTI_ROWCOPY:
            bucket_add(&s->ready_act[w->dimm][w->bank], wid);
            break;
        case WORK_FPGA_POP:
            bucket_add(&s->ready_fpga[0], wid);
            break;
        case WORK_FPGA_NL:
            bucket_add(&s->ready_fpga[1], wid);
            break;
        case WORK_FPGA_ATTN:
            bucket_add(&s->ready_fpga[2], wid);
            break;
        default:
            fprintf(stderr, "BUG: unknown work_type %d for wid %d\n", w->work_type, wid);
            break;
    }
}

static void update_ready(Scheduler *s, int completed_wid) {
    RevDeps *r = &rev_deps[completed_wid];
    for (int i = 0; i < r->n; i++) {
        int wid = r->deps[i];
        WorkItem *w = &s->work[wid];
        if (w->started || w->completed) continue;
        w->n_remaining_deps--;
        if (w->n_remaining_deps < 0) {
            fprintf(stderr, "BUG: n_remaining_deps went negative for wid %d\n", wid);
            exit(1);
        }
        if (w->n_remaining_deps == 0) {
            add_to_ready_bucket(s, wid);
        }
    }
}

/* ===================================================================
 * DDR4 timing checks
 * =================================================================== */

static int can_act(DIMMState *ds, int bank, int cycle) {
    DIMMConfig *c = &ds->cfg;
    if (ds->bank_busy[bank] > cycle) return 0;
    if (ds->cmd_busy > cycle) return 0;

    int bg = bank / c->banks_per_bg;

    if (ds->last_act_bg[bg] >= 0 && cycle - ds->last_act_bg[bg] < c->tRRD_L)
        return 0;

    for (int g = 0; g < c->n_bank_groups; g++) {
        if (g != bg && ds->last_act_bg[g] >= 0 &&
            cycle - ds->last_act_bg[g] < c->tRRD_S)
            return 0;
    }

    if (ds->n_acts >= 4) {
        int oldest = ds->last_4_acts[(ds->n_acts - 4) % 8];
        if (cycle - oldest < c->tFAW) return 0;
    }

    return 1;
}

/* ===================================================================
 * Issue functions
 * =================================================================== */

/* Issue an in-DRAM operation (MAJ3 or RowCopy).
 * n_acts_internal: MAJ3=3 (3-row APA), RowCopy=2 (2-row APA).
 * All internal ACTs count toward tFAW. Command bus blocked for APA sequence. */
static void issue_act_multi(DIMMState *ds, int bank, int cycle, int duration,
                             int n_acts_internal) {
    ds->bank_busy[bank] = cycle + duration;
    int apa_duration = 1;
    if (n_acts_internal == 3)
        apa_duration = 1 + ds->cfg.t_12_maj3 + 1 + ds->cfg.t_23_maj3
                     + 1 + ds->cfg.t_12_maj3 + 1 + ds->cfg.t_23_maj3 + 1;
    else if (n_acts_internal == 2)
        apa_duration = 1 + ds->cfg.t_12_rc + 1 + ds->cfg.t_23_rc + 1;
    ds->cmd_busy = cycle + apa_duration;
    int bg = bank / ds->cfg.banks_per_bg;
    ds->last_act_bg[bg] = cycle;
    for (int a = 0; a < n_acts_internal; a++) {
        ds->last_4_acts[ds->n_acts % 8] = cycle;
        ds->n_acts++;
    }
    ds->n_commands += n_acts_internal;
}

/* Issue a bus operation — accounts for bank-group interleaving when bg_parallel=1.
 *
 * Conservative model (bg_parallel=0): bus_busy blocks for full data transfer
 *   period (duration - tRP). This matches single-bank reads at tCCD_L spacing
 *   where data bus is 80% utilized (tBurst=4 out of tCCD_L=5).
 *
 * Bank-group-parallel model (bg_parallel=1): When multiple BGs alternate CAS
 *   commands at tCCD_S=4, the data bus operates at ~100% utilization.
 *   Per-read data bus time becomes COLS × tBurst instead of COLS × tCCD_L.
 *   Savings per read: COLS × (tCCD_L - tBurst) = COLS × 1 tCK.
 *   bus_busy still tracks serial contention (data bus is physically one),
 *   but with shorter occupancy per read. */
static void issue_bus_op(DIMMState *ds, int bank, int cycle, int duration) {
    ds->bank_busy[bank] = cycle + duration;
    ds->cmd_busy = cycle + 1;
    int bus_end = duration - ds->cfg.tRP;
    if (bus_end < ds->cfg.tBurst) bus_end = ds->cfg.tBurst;
    /* Default: bus_busy reflects conservative (same-BG CAS at tCCD_L). */
    ds->bus_busy = cycle + bus_end;
    int bg = bank / ds->cfg.banks_per_bg;
    ds->bus_busy_bg[bg] = cycle + bus_end;
    ds->last_act_bg[bg] = cycle;
    ds->last_4_acts[ds->n_acts % 8] = cycle;
    ds->n_acts++;
    ds->n_commands++;
}

/* Issue refresh — blocks all banks. */
static void issue_ref(DIMMState *ds, int cycle) {
    DIMMConfig *c = &ds->cfg;
    for (int b = 0; b < c->n_banks; b++)
        if (ds->bank_busy[b] < cycle + c->tRFC)
            ds->bank_busy[b] = cycle + c->tRFC;
    ds->bus_busy = cycle + c->tRFC;
    ds->cmd_busy = cycle + 1;
    ds->last_ref = cycle;
    ds->n_commands++;
}

/* ===================================================================
 * Work graph population — full decode token
 * =================================================================== */

/* Compute bus time for transferring 'nbytes' to/from one bank (page-mode).
 * Uses cas_interval (tCCD_L by default, tBurst when bg_parallel is on). */
static int bus_time(DIMMConfig *c, int nbytes, int is_write) {
    int cols = (nbytes + BURST_BYTES - 1) / BURST_BYTES;
    if (cols < 1) cols = 1;
    int t = c->tRCD + cols * c->cas_interval + c->tRP;
    if (is_write) t += c->tWR;
    return t;
}

/* Schedule one weight-row cycle (1 neuron group x 1 bitplane) on a bank.
 *
 * Protocol per weight-row-cycle:
 *   1. RowCopy activation backup -> working row (MAJ3 destroys activation)
 *   2. MAJ3(weight, activation_working, zero_pool_row) -> result
 *   3. Bus READ full row: 120 column accesses -> FPGA streams + popcounts
 *   4. RowCopy weight backup -> weight row (restore, if needed)
 */
static int sched_add_weight_row_cycle(Scheduler *s, int di, int bank,
                                       int wr, int bp, int act_dep,
                                       int barrier_wid) {
    DIMMConfig *c = &s->dimms[di].cfg;
    Layout *lay = &s->layout;

    /* Step 1a: RowCopy zeros (only needed without zero pool) */
    int step1a = -1;
    if (lay->zero_pool_size == 0) {
        step1a = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
        add_dep(s, act_dep, step1a);
    }

    /* Step 1b: RowCopy activation backup -> activation working row. */
    int step1b = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
    if (step1a >= 0)
        add_dep(s, step1a, step1b);
    else
        add_dep(s, act_dep, step1b);

    /* Step 2: MAJ3 + optional in-DRAM popcount.
     * If in-DRAM popcount: bank stays busy for MAJ3 + t_popcount (SA computes
     * popcount internally, no additional ACTs or bus needed). */
    int maj3_dur = c->maj3_time;
    if (lay->popcount_in_dram)
        maj3_dur += lay->t_popcount;
    int step2 = add_work(s, WORK_MAJ3, di, bank, maj3_dur);
    add_dep(s, step1b, step2);

    /* Step 3: Read result from DRAM to FPGA */
    int step3;
    if (lay->popcount_in_dram) {
        /* In-DRAM popcount already computed. Read just the result:
         * 12 neurons × 13-bit popcount = 20 bytes → 1 column (64 bytes) */
        int small_rd = bus_time(c, 64, 0);
        step3 = add_work(s, WORK_BUS_READ, di, bank, small_rd);
        add_dep(s, step2, step3);
    } else {
        /* FPGA popcount: must read full row (120 columns, 7680 bytes) */
        step3 = add_work(s, WORK_BUS_READ, di, bank, c->bus_rd_time);
        add_dep(s, step2, step3);
    }

    /* Step 4: Weight restore.
     * MAJ3 destroys the weight working row each bitplane. With weight_copies=N,
     * we have N pre-prepared copies; one is consumed per bitplane MAJ3.
     * After weight_copies bitplanes have all consumed their copies, we must
     * restore ALL N copies from the persistent backup.
     *
     * Restoration uses Multi-RowCopy (1 source -> N-1 destinations + source
     * preserved = N rows total). Validated by SiMRA's MultiRowInit test (N up to 31).
     * Bank-busy time is the same as a single RowCopy (~26 tCK) because the APA
     * sequence is identical; only the destination row addresses differ to
     * activate multiple destinations simultaneously.
     *
     * If weight_copies==1: restore is per-bitplane (1-to-1 RowCopy). */
    int step4 = step3;
    int need_restore = 1;
    if (lay->weight_copies > 1) {
        if ((bp + 1) % lay->weight_copies != 0 && bp < lay->n_bitplanes - 1)
            need_restore = 0;
    }
    if (need_restore) {
        if (lay->weight_copies > 1) {
            /* Multi-RowCopy: 1 source -> (weight_copies-1) destinations */
            step4 = add_work(s, WORK_MULTI_ROWCOPY, di, bank, c->rc_time);
        } else {
            step4 = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
        }
        add_dep(s, step3, step4);
    }

    /* Step 5: FPGA popcount (pipelined, overlaps with data arrival) */
    int pop = add_work(s, WORK_FPGA_POP, -1, -1, 4);
    add_dep(s, step3, pop);
    add_dep(s, pop, barrier_wid);

    return step4;
}

/* POPCNT3 variant: MAJ3 only (no bus read), result stays in bank.
 * Returns the MAJ3 item wid (for chaining into POPCNT3 reduction). */
static int sched_add_maj3_only(Scheduler *s, int di, int bank,
                                int bp, int act_dep) {
    DIMMConfig *c = &s->dimms[di].cfg;
    Layout *lay = &s->layout;

    int step1a = -1;
    if (lay->zero_pool_size == 0) {
        step1a = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
        add_dep(s, act_dep, step1a);
    }

    int step1b = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
    if (step1a >= 0) add_dep(s, step1a, step1b);
    else add_dep(s, act_dep, step1b);

    int step2 = add_work(s, WORK_MAJ3, di, bank, c->maj3_time);
    add_dep(s, step1b, step2);

    /* Weight restore — Multi-RowCopy when weight_copies > 1 (1 src -> N-1 dst),
     * single RowCopy otherwise. See same logic in sched_add_weight_row_cycle. */
    int step3 = step2;
    if (lay->weight_copies > 1) {
        if ((bp + 1) % lay->weight_copies != 0 && bp < lay->n_bitplanes - 1) {
            /* skip restore — copies still available */
        } else {
            step3 = add_work(s, WORK_MULTI_ROWCOPY, di, bank, c->rc_time);
            add_dep(s, step2, step3);
        }
    } else {
        step3 = add_work(s, WORK_ROWCOPY, di, bank, c->rc_time);
        add_dep(s, step2, step3);
    }

    return step2;  /* the MAJ3 item — POPCNT3 reduction depends on this */
}

/* Create shared activation write items for a group of projections that share
 * the same input (e.g., Q/K/V all use rmsnorm1 output). Returns a flat array
 * of act_backup wids indexed by [dimm][bp * n_banks + bank].
 * These can be passed to multiple projections to avoid redundant bus writes. */
#define MAX_SHARED_ACT (MAX_DIMMS * MAX_BANKS * MAX_BITPLANES)

static void sched_create_shared_activation(Scheduler *s, int prev_dep,
                                            int shared_act[MAX_SHARED_ACT]) {
    int n_dimms = s->n_dimms;
    int nbp = s->layout.n_bitplanes;
    int pb = s->layout.proj_banks;  /* banks that get bus writes */

    for (int di = 0; di < n_dimms; di++) {
        DIMMConfig *c = &s->dimms[di].cfg;
        int n_banks = c->n_banks;
        /* Effective banks needing activation = min(weight_banks, n_banks).
         * Only banks that will hold weight rows need activation data. */
        int wb = s->layout.weight_banks;
        if (wb <= 0 || wb > n_banks) wb = n_banks;
        int banks_need_act = wb;

        for (int bp = 0; bp < nbp; bp++) {
            for (int b = 0; b < banks_need_act; b++) {
                int ab;
                if (b < pb) {
                    /* Bus write to this bank (standard path) */
                    int aw = add_work(s, WORK_BUS_WRITE, di, b, c->bus_wr_time);
                    add_dep(s, prev_dep, aw);
                    ab = add_work(s, WORK_ROWCOPY, di, b, c->rc_time);
                    add_dep(s, aw, ab);
                } else if (s->layout.use_lisa) {
                    /* LISA: copy from bank 0's activation via cross-subarray link. */
                    int src = shared_act[di * nbp * MAX_BANKS + bp * n_banks + 0];
                    int lisa_time = c->rc_time * 2;
                    ab = add_work(s, WORK_ROWCOPY, di, b, lisa_time);
                    add_dep(s, src, ab);
                } else {
                    /* No LISA: must bus write to every used bank */
                    int aw = add_work(s, WORK_BUS_WRITE, di, b, c->bus_wr_time);
                    add_dep(s, prev_dep, aw);
                    ab = add_work(s, WORK_ROWCOPY, di, b, c->rc_time);
                    add_dep(s, aw, ab);
                }
                shared_act[di * nbp * MAX_BANKS + bp * n_banks + b] = ab;
            }
        }
    }
}

/* Bit-serial projection scheduler.
 *
 * Layout: each row = weights for ONE input dim across d_out output neurons at
 *   columns. Per-batch MAJ3+POPCNT3 accumulation across K input dims, giving
 *   per-column partial popcounts. Final FPGA sums partial popcounts across
 *   batches.
 *
 * Note: d_in determines weight row count; d_out determines per-row column usage.
 *   Both Q/K/V/O/gate/up have d_in=D_MODEL, down has d_in=D_FF, unembed has d_in=D_MODEL.
 *
 * Per batch of K inputs on ONE bank:
 *   1. K activation broadcast rows via RowCopy from templates (no bus)
 *   2. K MAJ3 operations (one per input dim, produces AND result rows)
 *   3. POPCNT3 tree: K - R reduction operations where R=ceil(log2(K+1)) result rows
 *   4. R bus reads (output the compressed result rows)
 * The FPGA aggregates partial popcounts across batches via shift-and-add.
 */
static int sched_add_projection_bitserial(Scheduler *s, const char *name,
                                           int d_in, int d_out, int prev_dep) {
    int n_dimms = s->n_dimms;
    int nbp = s->layout.n_bitplanes;
    int K = s->layout.bs_K;
    if (K <= 0) K = 127;

    int barrier = add_work(s, WORK_BARRIER, -1, -1, 0);

    /* Number of batches needed to cover d_in inputs */
    int batches_total = (d_in + K - 1) / K;

    DIMMConfig *c0 = &s->dimms[0].cfg;
    int maj3_time = c0->maj3_time;
    int rc_time = c0->rc_time;
    int t_popcnt3 = s->layout.t_popcnt3 > 0 ? s->layout.t_popcnt3 : 243;

    /* Compute result rows count for K inputs */
    int R = 1;
    { int tmp = K; while ((1 << R) <= tmp) R++; }

    /* Distribute batches across DIMMs */
    for (int di = 0; di < n_dimms; di++) {
        int batches_start = (batches_total * di) / n_dimms;
        int batches_end = (batches_total * (di + 1)) / n_dimms;
        int my_batches = batches_end - batches_start;
        if (my_batches == 0) continue;

        DIMMConfig *c = &s->dimms[di].cfg;
        int n_banks = c->n_banks;

        /* Each batch occupies one bank (subarray) for all its work.
         * Distribute batches × bitplanes across banks.
         * With my_batches * nbp total "work units" and n_banks available banks,
         * each bank processes (my_batches * nbp / n_banks) units. */
        for (int bp = 0; bp < nbp; bp++) {
            for (int batch_i = 0; batch_i < my_batches; batch_i++) {
                int bank = (batch_i * nbp + bp) % n_banks;

                /* 1. Activation broadcast — Multi-RowCopy from 2 templates
                 *    (all-zeros and all-ones), grouped by activation bit value.
                 *
                 *    Each k input element has activation bit ∈ {0,1}. Per group
                 *    of up to MRC_FANOUT (31) destinations sharing the same value,
                 *    one Multi-RowCopy op replicates that template into the
                 *    K_v destination working rows.
                 *
                 *    Worst-case op count: 2 * ceil(K / MRC_FANOUT) (assumes both
                 *    values present in every fanout group). This collapses K
                 *    sequential 1-to-1 RowCopies into ~K/16 ops on average.
                 *
                 *    Then K MAJ3 ops still execute, one per input element. */
                int n_mrc_groups = (K + MRC_FANOUT - 1) / MRC_FANOUT;
                int n_mrc_ops = 2 * n_mrc_groups;  /* one zero-template + one one-template per group */
                int last_mrc = -1;
                int first_mrc = -1;
                for (int m = 0; m < n_mrc_ops; m++) {
                    int mrc = add_work(s, WORK_MULTI_ROWCOPY, di, bank, rc_time);
                    if (last_mrc >= 0) add_dep(s, last_mrc, mrc);
                    else if (prev_dep >= 0) add_dep(s, prev_dep, mrc);
                    last_mrc = mrc;
                    if (first_mrc < 0) first_mrc = mrc;
                }

                int last_dep = (last_mrc >= 0) ? last_mrc : prev_dep;
                int first_maj3 = -1;
                for (int k = 0; k < K; k++) {
                    /* MAJ3 on (weight, activation_broadcast, zero).
                     * Activation broadcast row is now ready from Multi-RowCopy phase. */
                    int maj3 = add_work(s, WORK_MAJ3, di, bank, maj3_time);
                    add_dep(s, last_dep, maj3);

                    last_dep = maj3;
                    if (first_maj3 < 0) first_maj3 = maj3;
                }

                /* 2. POPCNT3 reduction: (K - R) sequential ops on this bank.
                 *    Each op uses MAJ3 + XOR3 (modeled as WORK_MAJ3 duration t_popcnt3). */
                int popcnt3_count = K - R;
                if (popcnt3_count < 0) popcnt3_count = 0;
                int reduce_dep = last_dep;
                if (popcnt3_count > 0) {
                    /* Model as single bank-busy block (sequential POPCNT3 ops) */
                    int reduce = add_work(s, WORK_MAJ3, di, bank,
                                          popcnt3_count * t_popcnt3);
                    add_dep(s, last_dep, reduce);
                    reduce_dep = reduce;
                }

                /* 3. Read R compressed result rows */
                for (int r = 0; r < R; r++) {
                    int rd = add_work(s, WORK_BUS_READ, di, bank, c->bus_rd_time);
                    add_dep(s, reduce_dep, rd);
                    int pop = add_work(s, WORK_FPGA_POP, -1, -1, 4);
                    add_dep(s, rd, pop);
                    add_dep(s, pop, barrier);
                }
            }
        }
    }

    if (s->debug_events > 0) {
        printf("  Projection '%s' (bit-serial): d_in=%d, d_out=%d, K=%d, R=%d, "
               "batches=%d, barrier_deps=%d\n",
               name, d_in, d_out, K, R, batches_total,
               s->work[barrier].n_remaining_deps);
    }

    track_milestone(name, -1, barrier);
    return barrier;
}

/* Schedule a projection using pre-created shared activation items.
 * If shared_act is NULL, creates its own (no sharing).
 * In bit-serial mode, dispatches to the bit-serial projection scheduler
 * (which assumes d_in = D_MODEL for most projections, D_FF for down). */
static int sched_add_projection_ex_din(Scheduler *s, const char *name,
                                        int d_in, int d_out, int prev_dep,
                                        int *shared_act) {
    if (s->layout.bit_serial) {
        return sched_add_projection_bitserial(s, name, d_in, d_out, prev_dep);
    }
    /* Neuron-packed path below */
    int n_wr = (d_out + NEURONS_PER_ROW - 1) / NEURONS_PER_ROW;
    int n_dimms = s->n_dimms;
    int nbp = s->layout.n_bitplanes;

    int barrier = add_work(s, WORK_BARRIER, -1, -1, 0);

    /* Local activation if not shared */
    int local_act[MAX_SHARED_ACT];
    if (!shared_act) {
        sched_create_shared_activation(s, prev_dep, local_act);
        shared_act = local_act;
    }

    for (int di = 0; di < n_dimms; di++) {
        DIMMConfig *c = &s->dimms[di].cfg;
        int n_banks = c->n_banks;

        int wr_start = (n_wr * di) / n_dimms;
        int wr_end = (n_wr * (di + 1)) / n_dimms;
        int my_n_wr = wr_end - wr_start;
        if (my_n_wr == 0) continue;

        /* Effective banks for weight row distribution (default: all n_banks) */
        int wb = s->layout.weight_banks;
        if (wb <= 0 || wb > n_banks) wb = n_banks;

        if (s->layout.use_popcnt3) {
            /* POPCNT3 mode: accumulate MAJ3 results per bank per bitplane,
             * then read compressed results.
             * Per bank: N weight rows per bitplane → N-ceil(log2(N+1)) POPCNT3 ops
             * → ceil(log2(N+1)) result rows to read.
             * Fewer banks => more rows/bank => better compression. */
            for (int bp = 0; bp < nbp; bp++) {
                for (int b = 0; b < wb; b++) {
                    int act_dep = shared_act[di * nbp * MAX_BANKS + bp * n_banks + b];

                    /* Count weight rows on this bank */
                    int n_on_bank = 0;
                    for (int wr_i = 0; wr_i < my_n_wr; wr_i++)
                        if (wr_i % wb == b) n_on_bank++;

                    if (n_on_bank == 0) continue;

                    /* Schedule all MAJ3s for this bank+bitplane (no bus read) */
                    int last_maj3 = -1;
                    for (int wr_i = 0; wr_i < my_n_wr; wr_i++) {
                        if (wr_i % wb != b) continue;
                        int m = sched_add_maj3_only(s, di, b, bp, act_dep);
                        last_maj3 = m;
                    }

                    /* POPCNT3 reduction phase: bank-only, no bus.
                     * N rows → ceil(log2(N+1)) result rows.
                     * Ops: N - ceil(log2(N+1)) POPCNT3, each t_popcnt3 tCK. */
                    int result_rows = 1;
                    { int tmp = n_on_bank; while ((1 << result_rows) <= tmp) result_rows++; }
                    int n_popcnt3_ops = n_on_bank - result_rows;
                    if (n_popcnt3_ops < 0) n_popcnt3_ops = 0;
                    int reduce_dur = n_popcnt3_ops * s->layout.t_popcnt3;

                    if (reduce_dur > 0) {
                        /* Model as single bank-busy block (conservative: sequential) */
                        int reduce = add_work(s, WORK_MAJ3, di, b, reduce_dur);
                        add_dep(s, last_maj3, reduce);
                        last_maj3 = reduce;
                    }

                    /* Read compressed result rows */
                    for (int r = 0; r < result_rows; r++) {
                        int rd = add_work(s, WORK_BUS_READ, di, b, c->bus_rd_time);
                        add_dep(s, last_maj3, rd);
                        int pop = add_work(s, WORK_FPGA_POP, -1, -1, 4);
                        add_dep(s, rd, pop);
                        add_dep(s, pop, barrier);
                    }
                }
            }
        } else {
            /* Standard mode: read each MAJ3 result immediately.
             * Weight rows distributed across `wb` banks (default: all n_banks). */
            for (int wr_i = 0; wr_i < my_n_wr; wr_i++) {
                int bank = wr_i % wb;
                for (int bp = 0; bp < nbp; bp++) {
                    int act_dep = shared_act[di * nbp * MAX_BANKS + bp * n_banks + bank];
                    sched_add_weight_row_cycle(s, di, bank, wr_start + wr_i,
                                               bp, act_dep, barrier);
                }
            }
        }
    }

    if (s->debug_events > 0) {
        printf("  Projection '%s': d_out=%d, n_wr=%d, barrier_deps=%d\n",
               name, d_out, n_wr, s->work[barrier].n_remaining_deps);
    }

    track_milestone(name, -1, barrier);
    return barrier;
}

/* Backward-compat wrapper: assume d_in = D_MODEL for most projections. */
static int sched_add_projection_ex(Scheduler *s, const char *name,
                                    int d_out, int prev_dep,
                                    int *shared_act) {
    return sched_add_projection_ex_din(s, name, D_MODEL, d_out, prev_dep, shared_act);
}

/* Convenience: projection with its own activation writes (no sharing). */
static int sched_add_projection(Scheduler *s, int di_ignored, const char *name,
                                 int d_out, int prev_dep) {
    return sched_add_projection_ex(s, name, d_out, prev_dep, NULL);
}

static int sched_add_fpga_op(Scheduler *s, int type, int duration, int dep) {
    int wid = add_work(s, type, -1, -1, duration);
    add_dep(s, dep, wid);
    return wid;
}

static int sched_add_fpga_op2(Scheduler *s, int type, int duration,
                               int dep1, int dep2) {
    int wid = add_work(s, type, -1, -1, duration);
    add_dep(s, dep1, wid);
    add_dep(s, dep2, wid);
    return wid;
}

static int sched_add_fpga_op3(Scheduler *s, int type, int duration,
                               int dep1, int dep2, int dep3) {
    int wid = add_work(s, type, -1, -1, duration);
    add_dep(s, dep1, wid);
    add_dep(s, dep2, wid);
    add_dep(s, dep3, wid);
    return wid;
}

/* ===================================================================
 * Cold-start phase modeling
 *
 * One-time initialization performed before any inference begins. Two phases:
 *
 *   1. Zero-pool population: each subarray needs `zero_pool_size` zero rows.
 *      Starting from one pre-cleared row, populate the rest via Multi-RowCopy.
 *      Per bank: ceil((zero_pool_size - 1) / MRC_FANOUT) Multi-RowCopy ops.
 *
 *   2. Weight-copy creation: each weight backup row holds NEURONS_PER_ROW (12)
 *      neurons. With weight_copies > 1, replicate each backup into N copies
 *      via Multi-RowCopy. Per backup row: ceil((weight_copies - 1) / MRC_FANOUT)
 *      ops. SiMRA validates fanout up to 31 destinations.
 *
 * Cold-start time is a ONE-TIME cost amortized across all decoded tokens.
 * Reported separately from per-token throughput.
 * =================================================================== */

static int sched_add_cold_start(Scheduler *s) {
    Layout *lay = &s->layout;
    if (lay->zero_pool_size <= 0 && lay->weight_copies <= 1)
        return -1;  /* Nothing to do */

    int barrier = add_work(s, WORK_BARRIER, -1, -1, 0);

    /* Per-layer weight backup row counts (neuron-packed layout):
     * Q,K,V,O,down at d_out=D_MODEL → ceil(D_MODEL/12) rows
     * gate,up at d_out=D_FF → ceil(D_FF/12) rows */
    int rows_d_model = (D_MODEL + NEURONS_PER_ROW - 1) / NEURONS_PER_ROW;
    int rows_d_ff = (D_FF + NEURONS_PER_ROW - 1) / NEURONS_PER_ROW;
    int rows_per_layer = 5 * rows_d_model + 2 * rows_d_ff;  /* Q,K,V,O,down + gate,up */
    int rows_unembed = (VOCAB_SIZE + NEURONS_PER_ROW - 1) / NEURONS_PER_ROW;
    int total_weight_rows = N_LAYERS * rows_per_layer + rows_unembed;

    int n_dimms = s->n_dimms;
    int per_dimm_weight_rows = (total_weight_rows + n_dimms - 1) / n_dimms;

    /* Multi-RowCopies per weight backup row to populate (weight_copies - 1) destinations */
    int wc_extras = lay->weight_copies > 1 ? lay->weight_copies - 1 : 0;
    int mrc_per_weight_row = (wc_extras + MRC_FANOUT - 1) / MRC_FANOUT;

    /* Multi-RowCopies per bank for zero-pool: 1 source -> (zero_pool_size - 1) dests */
    int zp_extras = lay->zero_pool_size > 0 ? lay->zero_pool_size - 1 : 0;
    int mrc_per_bank_zp = (zp_extras + MRC_FANOUT - 1) / MRC_FANOUT;

    for (int di = 0; di < n_dimms; di++) {
        DIMMConfig *c = &s->dimms[di].cfg;
        int n_banks = c->n_banks;

        /* Zero-pool population: one Multi-RowCopy chain per bank */
        for (int b = 0; b < n_banks; b++) {
            int prev = -1;
            for (int m = 0; m < mrc_per_bank_zp; m++) {
                int wid = add_work(s, WORK_MULTI_ROWCOPY, di, b, c->rc_time);
                if (prev >= 0) add_dep(s, prev, wid);
                prev = wid;
            }
            if (prev >= 0) add_dep(s, prev, barrier);
        }

        /* Weight-copy creation: distribute backup rows across this DIMM's banks */
        if (mrc_per_weight_row > 0) {
            int rows_this_dimm = per_dimm_weight_rows;
            int per_bank = (rows_this_dimm + n_banks - 1) / n_banks;
            for (int b = 0; b < n_banks; b++) {
                int rows_on_bank = per_bank;
                if (b * per_bank >= rows_this_dimm) rows_on_bank = 0;
                else if ((b + 1) * per_bank > rows_this_dimm)
                    rows_on_bank = rows_this_dimm - b * per_bank;

                int prev = -1;
                for (int r = 0; r < rows_on_bank; r++) {
                    for (int m = 0; m < mrc_per_weight_row; m++) {
                        int wid = add_work(s, WORK_MULTI_ROWCOPY, di, b, c->rc_time);
                        if (prev >= 0) add_dep(s, prev, wid);
                        prev = wid;
                    }
                }
                if (prev >= 0) add_dep(s, prev, barrier);
            }
        }
    }

    if (s->debug_events > 0) {
        printf("Cold-start: zero_pool=%d (mrc/bank=%d), weight_copies=%d (mrc/row=%d), "
               "weight rows=%d total, %d per DIMM\n",
               lay->zero_pool_size, mrc_per_bank_zp, lay->weight_copies,
               mrc_per_weight_row, total_weight_rows, per_dimm_weight_rows);
    }

    return barrier;
}

static int sched_add_full_layer(Scheduler *s, int layer, int seq_len,
                                 FPGAConfig *fpga, int prev_dep) {
    int di = 0;
    DIMMConfig *c0 = &s->dimms[di].cfg;
    int vec_wr_time = bus_time(c0, D_MODEL, 1);

    /* RMSNorm 1: FPGA or pLUTo (in-DRAM) */
    int rmsnorm1;
    if (s->layout.use_pluto) {
        /* pLUTo: RMSNorm via in-DRAM LUT query. ~512 entries, no bus needed.
         * pLUTo paper: query time = tRCD × N_entries. For 512 entries: ~6400 tCK.
         * But RMSNorm also requires reduction (sum of squares) which pLUTo can't do.
         * Model conservatively: FPGA does reduction, pLUTo does element-wise normalization.
         * Net: same FPGA time for reduction, but the normalized result stays in DRAM. */
        rmsnorm1 = sched_add_fpga_op(s, WORK_FPGA_NL, fpga->rmsnorm_lat, prev_dep);
    } else {
        rmsnorm1 = sched_add_fpga_op(s, WORK_FPGA_NL, fpga->rmsnorm_lat, prev_dep);
    }

    /* Q, K, V share the same input activation */
    int shared_qkv[MAX_SHARED_ACT];
    sched_create_shared_activation(s, rmsnorm1, shared_qkv);
    int q = sched_add_projection_ex(s, "Q", D_MODEL, rmsnorm1, shared_qkv);
    int k = sched_add_projection_ex(s, "K", D_MODEL, rmsnorm1, shared_qkv);
    int v = sched_add_projection_ex(s, "V", D_MODEL, rmsnorm1, shared_qkv);

    /* Vector writebacks: FPGA accumulates cross-bank partial sums,
     * writes result back to DRAM. With pLUTo, the writeback also includes
     * bit-decomposition in-DRAM, but the bus write still happens. */
    int q_wb = add_work(s, WORK_BUS_WRITE, di, 0, vec_wr_time);
    add_dep(s, q, q_wb);
    int k_wb = add_work(s, WORK_BUS_WRITE, di, 1, vec_wr_time);
    add_dep(s, k, k_wb);
    int v_wb = add_work(s, WORK_BUS_WRITE, di, 2, vec_wr_time);
    add_dep(s, v, v_wb);

    int rope = sched_add_fpga_op3(s, WORK_FPGA_NL, fpga->rope_lat, q_wb, k_wb, v_wb);

    /* KV cache store */
    int kv_cols = (D_MODEL + BURST_BYTES - 1) / BURST_BYTES;
    int kv_wr_time = c0->tRCD + kv_cols * c0->cas_interval + c0->tWR + c0->tRP;
    int kv_rd_time = c0->tRCD + kv_cols * c0->cas_interval + c0->tRP;

    int kv_store_k = add_work(s, WORK_BUS_WRITE, di, 14, kv_wr_time);
    add_dep(s, rope, kv_store_k);
    int kv_store_v = add_work(s, WORK_BUS_WRITE, di, 15, kv_wr_time);
    add_dep(s, rope, kv_store_v);

    /* KV cache read */
    int vecs_per_row = ROW_BITS / (D_MODEL * 8);
    if (vecs_per_row < 1) vecs_per_row = 1;
    int kv_rows = (seq_len + vecs_per_row - 1) / vecs_per_row;
    if (kv_rows < 1) kv_rows = 1;

    int last_kv_rd = -1;
    for (int r = 0; r < kv_rows; r++) {
        int b = r % c0->n_banks;
        int rd = add_work(s, WORK_BUS_READ, di, b, kv_rd_time);
        add_dep(s, kv_store_k, rd);
        last_kv_rd = rd;
    }
    for (int r = 0; r < kv_rows; r++) {
        int b = (r + kv_rows) % c0->n_banks;
        int rd = add_work(s, WORK_BUS_READ, di, b, kv_rd_time);
        add_dep(s, kv_store_v, rd);
        last_kv_rd = rd;
    }

    /* Attention */
    int attn_dur = seq_len * N_HEADS * fpga->attn_dot_lat;
    int attn_qkt = add_work(s, WORK_FPGA_ATTN, -1, -1, attn_dur);
    add_dep(s, last_kv_rd, attn_qkt);
    add_dep(s, q_wb, attn_qkt);
    int attn_sm = sched_add_fpga_op(s, WORK_FPGA_NL, fpga->softmax_lat * N_HEADS, attn_qkt);
    int attn_sv_dur = seq_len * N_HEADS * fpga->attn_sv_lat;
    int attn_sv = sched_add_fpga_op(s, WORK_FPGA_ATTN, attn_sv_dur, attn_sm);

    /* O projection */
    int o = sched_add_projection(s, 0, "O", D_MODEL, attn_sv);
    int o_wb = add_work(s, WORK_BUS_WRITE, di, 0, vec_wr_time);
    add_dep(s, o, o_wb);

    int res1 = sched_add_fpga_op(s, WORK_FPGA_NL, 2, o_wb);
    int rmsnorm2 = sched_add_fpga_op(s, WORK_FPGA_NL, fpga->rmsnorm_lat, res1);

    /* FFN: gate and up share the same input (rmsnorm2 output) */
    int ffn_vec_wr = bus_time(c0, D_FF, 1);
    int shared_ffn[MAX_SHARED_ACT];
    sched_create_shared_activation(s, rmsnorm2, shared_ffn);
    int gate = sched_add_projection_ex(s, "gate", D_FF, rmsnorm2, shared_ffn);
    int gate_wb = add_work(s, WORK_BUS_WRITE, di, 0, ffn_vec_wr);
    add_dep(s, gate, gate_wb);

    int up = sched_add_projection_ex(s, "up", D_FF, rmsnorm2, shared_ffn);
    int up_wb = add_work(s, WORK_BUS_WRITE, di, 1, ffn_vec_wr);
    add_dep(s, up, up_wb);

    int silu = sched_add_fpga_op2(s, WORK_FPGA_NL, fpga->silu_lat, gate_wb, up_wb);
    int gxu = sched_add_fpga_op(s, WORK_FPGA_NL, 10, silu);
    int mid_wb = add_work(s, WORK_BUS_WRITE, di, 0, ffn_vec_wr);
    add_dep(s, gxu, mid_wb);

    /* down projection: d_in=D_FF, d_out=D_MODEL. In bit-serial mode, d_in matters. */
    int down = sched_add_projection_ex_din(s, "down", D_FF, D_MODEL, mid_wb, NULL);
    int down_wb = add_work(s, WORK_BUS_WRITE, di, 0, vec_wr_time);
    add_dep(s, down, down_wb);

    int res2 = sched_add_fpga_op(s, WORK_FPGA_NL, 2, down_wb);
    return res2;
}

/* ===================================================================
 * Main scheduler loop — resource-first slot filling
 * =================================================================== */

static void sched_run(Scheduler *s) {
    int max_cycles = 2000000000;
    int progress_interval = 100000;
    clock_t last_report = clock();

    printf("Scheduler running: %d items, %d initially ready\n",
           s->n_work, s->n_ready);
    fflush(stdout);

    while (s->n_completed < s->n_work && s->cycle < max_cycles) {

        /* Phase 1: Process completions */
        while (heap_peek_cycle(&s->inflight) <= s->cycle) {
            HeapEntry e = heap_pop(&s->inflight);
            WorkItem *w = &s->work[e.wid];
            if (w->completed) continue;
            w->completed = 1;
            s->n_completed++;
            if (s->debug_events > 0 && s->debug_count < s->debug_events) {
                printf("  [%d] DONE wid=%d type=%s dimm=%d bank=%d\n",
                       s->cycle, e.wid, work_type_name[w->work_type], w->dimm, w->bank);
                s->debug_count++;
            }
            update_ready(s, e.wid);
        }

        /* Phase 2: Refresh */
        for (int d = 0; d < s->n_dimms; d++) {
            DIMMState *ds = &s->dimms[d];
            if (s->cycle - ds->last_ref >= ds->cfg.tREFI) {
                int can_ref = 1;
                for (int b = 0; b < ds->cfg.n_banks; b++)
                    if (ds->bank_busy[b] > s->cycle) { can_ref = 0; break; }
                if (can_ref) issue_ref(ds, s->cycle);
            }
        }

        /* Phase 3: Issue DRAM work */
        int any_dram_issued = 0;
        int any_fpga_issued = 0;

        for (int d = 0; d < s->n_dimms; d++) {
            DIMMState *ds = &s->dimms[d];
            if (ds->cmd_busy > s->cycle) continue;

            int issued = 0;

            /* Priority 1: Bus operations.
             * With bg_parallel=1: shorter per-read time (tBurst cadence) reflects
             * the sustained throughput when scheduler alternates BGs. The data bus
             * is still serialized (physically one channel) — we don't model 4×
             * parallel. This gives a realistic ~1.2× speedup, not an unrealistic 4×. */
            if (!issued && ds->bus_busy <= s->cycle) {
                for (int b = 0; b < ds->cfg.n_banks && !issued; b++) {
                    if (s->ready_bus[d][b].n == 0) continue;
                    if (!can_act(ds, b, s->cycle)) continue;

                    int wid = bucket_pop_min(&s->ready_bus[d][b]);
                    while (wid >= 0 && s->work[wid].started) {
                        s->n_ready--;
                        wid = bucket_pop_min(&s->ready_bus[d][b]);
                    }
                    if (wid < 0) continue;

                    WorkItem *w = &s->work[wid];
                    issue_bus_op(ds, w->bank, s->cycle, w->duration);
                    w->started = 1;
                    w->start_cycle = s->cycle;
                    w->end_cycle = s->cycle + w->duration;
                    s->n_ready--;
                    heap_push(&s->inflight, w->end_cycle, wid);
                    issued = 1;
                    any_dram_issued = 1;
                }
            }

            /* Priority 2: In-DRAM ops */
            if (!issued) {
                for (int b = 0; b < ds->cfg.n_banks && !issued; b++) {
                    if (s->ready_act[d][b].n == 0) continue;
                    if (!can_act(ds, b, s->cycle)) continue;

                    int wid = bucket_pop_min(&s->ready_act[d][b]);
                    while (wid >= 0 && s->work[wid].started) {
                        s->n_ready--;
                        wid = bucket_pop_min(&s->ready_act[d][b]);
                    }
                    if (wid < 0) continue;

                    WorkItem *w = &s->work[wid];
                    int n_int_acts = (w->work_type == WORK_MAJ3) ? 3 : 2;
                    issue_act_multi(ds, w->bank, s->cycle, w->duration, n_int_acts);
                    w->started = 1;
                    w->start_cycle = s->cycle;
                    w->end_cycle = s->cycle + w->duration;
                    s->n_ready--;
                    heap_push(&s->inflight, w->end_cycle, wid);
                    issued = 1;
                    any_dram_issued = 1;
                }
            }
        }

        /* Phase 4: FPGA work */
        int fpga_busy[3];
        fpga_busy[0] = s->fpga_pop_busy;
        fpga_busy[1] = s->fpga_nl_busy;
        fpga_busy[2] = s->fpga_attn_busy;

        for (int f = 0; f < 3; f++) {
            if (fpga_busy[f] > s->cycle) continue;
            if (s->ready_fpga[f].n == 0) continue;

            int wid = bucket_pop_min(&s->ready_fpga[f]);
            while (wid >= 0 && s->work[wid].started) {
                s->n_ready--;
                wid = bucket_pop_min(&s->ready_fpga[f]);
            }
            if (wid < 0) continue;

            WorkItem *w = &s->work[wid];
            w->started = 1;
            w->start_cycle = s->cycle;
            w->end_cycle = s->cycle + w->duration;
            s->n_ready--;
            heap_push(&s->inflight, w->end_cycle, wid);

            if (f == 0) s->fpga_pop_busy = w->end_cycle;
            else if (f == 1) s->fpga_nl_busy = w->end_cycle;
            else s->fpga_attn_busy = w->end_cycle;
            any_fpga_issued = 1;
        }

        if (!any_dram_issued && !any_fpga_issued) s->total_nops++;

        /* Phase 5: Advance to next event */
        int next = INT_MAX;

        if (s->inflight.n > 0) next = heap_peek_cycle(&s->inflight);

        for (int d = 0; d < s->n_dimms; d++) {
            DIMMState *ds = &s->dimms[d];
            for (int b = 0; b < ds->cfg.n_banks; b++)
                if (ds->bank_busy[b] > s->cycle && ds->bank_busy[b] < next)
                    next = ds->bank_busy[b];
            if (ds->bus_busy > s->cycle && ds->bus_busy < next)
                next = ds->bus_busy;
            if (ds->cmd_busy > s->cycle && ds->cmd_busy < next)
                next = ds->cmd_busy;
        }

        if (s->fpga_pop_busy > s->cycle && s->fpga_pop_busy < next)
            next = s->fpga_pop_busy;
        if (s->fpga_nl_busy > s->cycle && s->fpga_nl_busy < next)
            next = s->fpga_nl_busy;
        if (s->fpga_attn_busy > s->cycle && s->fpga_attn_busy < next)
            next = s->fpga_attn_busy;

        for (int d = 0; d < s->n_dimms; d++) {
            int ref_due = s->dimms[d].last_ref + s->dimms[d].cfg.tREFI;
            if (ref_due > s->cycle && ref_due < next)
                next = ref_due;
        }

        if (next == INT_MAX || next <= s->cycle) {
            if (s->n_completed < s->n_work && s->n_ready == 0 && s->inflight.n == 0) {
                printf("\nDEADLOCK at cycle %d: %d/%d complete\n",
                       s->cycle, s->n_completed, s->n_work);
                int shown = 0;
                for (int i = 0; i < s->n_work && shown < 20; i++) {
                    WorkItem *w = &s->work[i];
                    if (!w->completed) {
                        printf("  wid=%d type=%s deps=%d started=%d\n",
                               i, work_type_name[w->work_type],
                               w->n_remaining_deps, w->started);
                        shown++;
                    }
                }
                break;
            }
            next = s->cycle + 1;
        }

        if (s->cycle > 0 && s->cycle % progress_interval == 0) {
            clock_t now = clock();
            double elapsed = (double)(now - last_report) / CLOCKS_PER_SEC;
            printf("  cycle %d: %d/%d done (%.1f%%), ready=%d, inflight=%d [%.2fs]\n",
                   s->cycle, s->n_completed, s->n_work,
                   100.0 * s->n_completed / s->n_work,
                   s->n_ready, s->inflight.n, elapsed);
            fflush(stdout);
            last_report = now;
        }

        s->cycle = next;
    }

    if (s->n_completed < s->n_work) {
        printf("\nWARNING: %d/%d items incomplete\n",
               s->n_work - s->n_completed, s->n_work);
    }
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(int argc, char **argv) {
    int n_layers_run = 1;
    int n_dimms = 1;
    int seq_len = 5;
    int debug_events = 0;
    int act_bits = 8;
    int popcount_in_dram = 0;
    int t_popcount = 20;
    int use_popcnt3 = 0;
    int t_popcnt3 = 243;
    int use_lisa = 0;
    int proj_banks = 0;   /* 0 = use all n_banks */
    int use_pluto = 0;
    int weight_banks = 0; /* 0 = use all n_banks */
    int bg_parallel = 0;  /* 0 = serialize bus ops DIMM-global; 1 = per-BG parallel */
    int bit_serial = 0;   /* 0 = neuron-packed (current); 1 = bit-serial layout */
    int bs_K = 127;       /* bit-serial batch size */
    int cold_start = 0;   /* 0 = skip; 1 = model & report cold-start init phase */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc)
            n_layers_run = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dimms") == 0 && i + 1 < argc)
            n_dimms = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc)
            seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--debug") == 0 && i + 1 < argc)
            debug_events = atoi(argv[++i]);
        else if (strcmp(argv[i], "--act-bits") == 0 && i + 1 < argc)
            act_bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--popcount") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "dram") == 0) popcount_in_dram = 1;
            else if (strcmp(argv[i], "fpga") == 0) popcount_in_dram = 0;
            else { fprintf(stderr, "--popcount must be 'fpga' or 'dram'\n"); return 1; }
        }
        else if (strcmp(argv[i], "--t-popcount") == 0 && i + 1 < argc)
            t_popcount = atoi(argv[++i]);
        else if (strcmp(argv[i], "--popcnt3") == 0)
            use_popcnt3 = 1;
        else if (strcmp(argv[i], "--t-popcnt3") == 0 && i + 1 < argc)
            t_popcnt3 = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lisa") == 0)
            use_lisa = 1;
        else if (strcmp(argv[i], "--proj-banks") == 0 && i + 1 < argc)
            proj_banks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pluto") == 0)
            use_pluto = 1;
        else if (strcmp(argv[i], "--weight-banks") == 0 && i + 1 < argc)
            weight_banks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bg-parallel") == 0)
            bg_parallel = 1;
        else if (strcmp(argv[i], "--bit-serial") == 0)
            bit_serial = 1;
        else if (strcmp(argv[i], "--bs-k") == 0 && i + 1 < argc)
            bs_K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cold-start") == 0)
            cold_start = 1;
    }

    if (act_bits < 1 || act_bits > MAX_BITPLANES) {
        fprintf(stderr, "act-bits must be 1-%d\n", MAX_BITPLANES);
        return 1;
    }

    /* LISA: reduce proj_banks to 1 (write activation to 1 bank, LISA distributes) */
    if (use_lisa && proj_banks == 0) proj_banks = 1;
    if (proj_banks == 0) proj_banks = 16;  /* default: all banks */

    Layout layout = { 16, 8, act_bits, popcount_in_dram, t_popcount,
                       use_popcnt3, t_popcnt3, use_lisa, proj_banks, use_pluto,
                       weight_banks, bg_parallel, bit_serial, bs_K };

    /* bg_parallel: with bank-group interleaving, DDR4 data bus achieves 100% use
     * (tBurst cadence) vs 80% for single-BG serial (tCCD_L cadence).
     * Effective per-read data time: COLS × tBurst instead of COLS × tCCD_L.
     * Recompute bus_rd_time and bus_wr_time with tBurst cadence. */
    if (bg_parallel) {
        /* Must recompute DIMM configs — but they're computed at sched_init time.
         * We set a flag and let the scheduler/DIMM init handle it below. */
    }
    FPGAConfig fpga = default_fpga();

    Scheduler s;
    sched_init(&s, n_dimms, layout);
    s.debug_events = debug_events;

    double tck_ns = 1.5;  /* DRAM Bender: DDR4 @ 1333 MT/s, CK=667 MHz */

    printf("=== CaSA Scheduler v2 ===\n");
    printf("Model: d=%d, d_ff=%d, heads=%d, layers=%d, vocab=%d\n",
           D_MODEL, D_FF, N_HEADS, n_layers_run, VOCAB_SIZE);
    printf("Layout: zero_pool=%d, weight_copies=%d\n",
           layout.zero_pool_size, layout.weight_copies);
    printf("Neurons/row: %d, act-bits: %d (bitplanes: %d)\n",
           NEURONS_PER_ROW, layout.n_bitplanes, layout.n_bitplanes);
    printf("DIMMs: %d, seq_len: %d, tCK: %.3f ns (DDR4-%d)\n",
           n_dimms, seq_len, tck_ns, (int)(2.0 / tck_ns * 1000));
    printf("Timing: tRCD=%d tRP=%d tRRD_S=%d tRRD_L=%d tFAW=%d tCCD_L=%d tBurst=%d\n",
           s.dimms[0].cfg.tRCD, s.dimms[0].cfg.tRP,
           s.dimms[0].cfg.tRRD_S, s.dimms[0].cfg.tRRD_L,
           s.dimms[0].cfg.tFAW, s.dimms[0].cfg.tCCD_L, s.dimms[0].cfg.tBurst);
    printf("Charge-sharing: t12=%d t23=%d tCK (%.1f/%.1f ns)\n",
           s.dimms[0].cfg.t_12_maj3, s.dimms[0].cfg.t_23_maj3,
           s.dimms[0].cfg.t_12_maj3 * tck_ns, s.dimms[0].cfg.t_23_maj3 * tck_ns);
    if (layout.use_popcnt3)
        printf("Accumulation: POPCNT3 (t_popcnt3=%d tCK = %.0f ns, FCDRAM NOT + SiMRA MAJ3)\n",
               layout.t_popcnt3, layout.t_popcnt3 * tck_ns);
    if (layout.use_lisa)
        printf("LISA: [HYPOTHETICAL] cross-subarray activation distribution, proj_banks=%d\n",
               layout.proj_banks);
    if (layout.use_pluto)
        printf("pLUTo: [HYPOTHETICAL] in-DRAM nonlinear (SiLU/RMSNorm via LUT, 10-23%% die area)\n");
    printf("Proj banks: %d (activation writes per bitplane per DIMM)\n", layout.proj_banks);
    {
        int wb = layout.weight_banks;
        if (wb <= 0 || wb > 16) wb = 16;
        printf("Weight banks: %d (weight row distribution per DIMM; fewer=more POPCNT3 compression)\n", wb);
    }
    printf("Popcount: %s", layout.popcount_in_dram ? "IN-DRAM" : "FPGA");
    if (layout.popcount_in_dram)
        printf(" (t_popcount=%d tCK = %.1f ns)", layout.t_popcount, layout.t_popcount * tck_ns);
    printf("\n");
    printf("Derived: MAJ3=%d tCK, RC=%d tCK, bus_rd=%d tCK, bus_wr=%d tCK\n",
           s.dimms[0].cfg.maj3_time, s.dimms[0].cfg.rc_time,
           s.dimms[0].cfg.bus_rd_time, s.dimms[0].cfg.bus_wr_time);
    fflush(stdout);

    /* === Build work graph === */
    clock_t t0 = clock();

    int cold_start_barrier = -1;
    int n_work_before_cold = 0;
    if (cold_start) {
        printf("Building cold-start phase...\n");
        fflush(stdout);
        cold_start_barrier = sched_add_cold_start(&s);
        n_work_before_cold = s.n_work;
        printf("  Cold-start items: %d\n", s.n_work);
    }

    int emb_rd_time = bus_time(&s.dimms[0].cfg, D_MODEL, 0);
    int emb = add_work(&s, WORK_BUS_READ, 0, 0, emb_rd_time);
    /* If we modeled cold-start, the first inference op cannot start until
     * cold-start completes. */
    if (cold_start_barrier >= 0)
        add_dep(&s, cold_start_barrier, emb);

    int prev = emb;
    for (int l = 0; l < n_layers_run; l++) {
        printf("Building layer %d/%d (items so far: %d)...\n", l+1, n_layers_run, s.n_work);
        fflush(stdout);
        prev = sched_add_full_layer(&s, l, seq_len, &fpga, prev);
    }

    int unembed = sched_add_projection(&s, 0, "unembed", VOCAB_SIZE, prev);
    (void)unembed;

    clock_t t1 = clock();
    double build_time = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("\nWork graph: %d items (built in %.2fs)\n", s.n_work, build_time);

    int type_counts[9] = {0};
    for (int i = 0; i < s.n_work; i++)
        type_counts[s.work[i].work_type]++;
    printf("  MAJ3: %d | RC: %d | BUS_RD: %d | BUS_WR: %d | "
           "FPGA_POP: %d | FPGA_NL: %d | FPGA_ATTN: %d | BARRIER: %d | MULTI_RC: %d\n",
           type_counts[0], type_counts[1], type_counts[2], type_counts[3],
           type_counts[4], type_counts[5], type_counts[6], type_counts[7], type_counts[8]);

    finalize_ready(&s);
    printf("Initially ready: %d\n", s.n_ready);
    fflush(stdout);

    /* === Run scheduler === */
    printf("\n");
    t0 = clock();
    sched_run(&s);
    t1 = clock();

    double sim_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double total_ns = (double)s.cycle * tck_ns;
    double total_us = total_ns / 1000.0;
    double total_ms = total_us / 1000.0;
    double tok_s = (total_us > 0) ? 1.0 / (total_us * 1e-6) : 0;

    /* Cold-start time: end of the cold-start barrier item, if modeled. */
    int cold_start_cycles = 0;
    if (cold_start_barrier >= 0) {
        cold_start_cycles = s.work[cold_start_barrier].end_cycle;
    }
    int decode_cycles = s.cycle - cold_start_cycles;
    double decode_ns = (double)decode_cycles * tck_ns;
    double decode_us = decode_ns / 1000.0;
    double decode_ms = decode_us / 1000.0;
    double decode_tok_s = (decode_us > 0) ? 1.0 / (decode_us * 1e-6) : 0;

    printf("\n=== Results (1 decode token) ===\n");
    if (cold_start_barrier >= 0) {
        double cs_ms = cold_start_cycles * tck_ns / 1e6;
        printf("Cold-start cycles: %d (%.2f ms, one-time amortized cost)\n",
               cold_start_cycles, cs_ms);
        printf("Cold-start items: %d (Multi-RowCopy ops)\n", n_work_before_cold);
        printf("Steady-state cycles: %d (%.1f us = %.2f ms)\n",
               decode_cycles, decode_us, decode_ms);
        printf("Steady-state decode throughput: %.2f tok/s\n", decode_tok_s);
    }
    printf("Total cycles: %d (%.1f us = %.2f ms)\n", s.cycle, total_us, total_ms);
    printf("Decode throughput (incl. cold-start): %.2f tok/s\n", tok_s);
    printf("Completed: %d/%d\n", s.n_completed, s.n_work);
    printf("Sim wall time: %.2fs\n", sim_time);

    /* Work breakdown */
    long long time_by_type[9] = {0};
    int count_by_type[9] = {0};
    for (int i = 0; i < s.n_work; i++) {
        WorkItem *w = &s.work[i];
        if (!w->completed) continue;
        count_by_type[w->work_type]++;
        time_by_type[w->work_type] += w->duration;
    }
    printf("\n=== Work Breakdown ===\n");
    printf("  %-10s  %8s  %12s  %8s\n", "Type", "Count", "Agg tCK", "Avg tCK");
    for (int t = 0; t < 9; t++) {
        if (count_by_type[t] == 0) continue;
        printf("  %-10s  %8d  %12lld  %8.1f\n",
               work_type_name[t], count_by_type[t], time_by_type[t],
               (double)time_by_type[t] / count_by_type[t]);
    }

    /* Bus analysis */
    long long bus_rd_tck = 0, bus_wr_tck = 0;
    long long bytes_rd = 0, bytes_wr = 0;
    int col_interval = s.dimms[0].cfg.cas_interval;
    for (int i = 0; i < s.n_work; i++) {
        WorkItem *w = &s.work[i];
        if (!w->completed) continue;
        if (w->work_type == WORK_BUS_READ || w->work_type == WORK_BUS_WRITE) {
            int overhead = s.dimms[0].cfg.tRCD + s.dimms[0].cfg.tRP;
            if (w->work_type == WORK_BUS_WRITE) overhead += s.dimms[0].cfg.tWR;
            int data_tck = w->duration - overhead;
            if (data_tck < col_interval) data_tck = col_interval;
            int cols = data_tck / col_interval;
            long long op_bytes = (long long)cols * BURST_BYTES;
            if (w->work_type == WORK_BUS_READ) {
                bus_rd_tck += data_tck;
                bytes_rd += op_bytes;
            } else {
                bus_wr_tck += data_tck;
                bytes_wr += op_bytes;
            }
        }
    }
    long long total_bus_tck = bus_rd_tck + bus_wr_tck;
    printf("\n=== Bus Analysis ===\n");
    printf("  Bus time: read=%lld + write=%lld = %lld tCK (%.1f%% of total)\n",
           bus_rd_tck, bus_wr_tck, total_bus_tck,
           100.0 * total_bus_tck / s.cycle);
    printf("  Data volume: read=%.1f MB, write=%.1f MB, total=%.1f MB\n",
           bytes_rd / 1e6, bytes_wr / 1e6, (bytes_rd + bytes_wr) / 1e6);
    double bw_used = (bytes_rd + bytes_wr) / (total_ms / 1000.0) / 1e9;
    double peak_bw = (double)BURST_BYTES / s.dimms[0].cfg.tBurst / (tck_ns * 1e-9) / 1e9;
    double practical_peak = peak_bw * (double)s.dimms[0].cfg.tBurst / col_interval;
    printf("  Effective BW: %.2f GB/s per channel (peak: %.1f, practical: %.1f GB/s)\n",
           bw_used, peak_bw, practical_peak);
    double min_bus_ms = (bytes_rd + bytes_wr) / 1e6 / practical_peak;
    printf("  Theoretical min: %.2f ms (actual: %.2f ms, efficiency: %.1f%%)\n",
           min_bus_ms, total_ms, 100.0 * min_bus_ms / total_ms);

    /* Utilization */
    printf("\n=== Utilization ===\n");
    for (int d = 0; d < n_dimms; d++) {
        long long bank_busy_total = 0, bus_busy = 0;
        for (int i = 0; i < s.n_work; i++) {
            WorkItem *w = &s.work[i];
            if (!w->completed || w->dimm != d) continue;
            if (w->work_type <= WORK_BUS_WRITE) bank_busy_total += w->duration;
            if (w->work_type == WORK_BUS_READ || w->work_type == WORK_BUS_WRITE) {
                int dt = w->duration - s.dimms[d].cfg.tRCD - s.dimms[d].cfg.tRP;
                if (w->work_type == WORK_BUS_WRITE) dt -= s.dimms[d].cfg.tWR;
                if (dt < 0) dt = s.dimms[d].cfg.tBurst;
                bus_busy += dt;
            }
        }
        printf("DIMM %d: bank=%.1f%%, bus=%.1f%%, cmds=%d\n", d,
               100.0 * bank_busy_total / (s.cycle * (double)s.dimms[d].cfg.n_banks),
               100.0 * bus_busy / s.cycle, s.dimms[d].n_commands);
    }
    {
        long long fp = 0, fn = 0, fa = 0;
        for (int i = 0; i < s.n_work; i++) {
            WorkItem *w = &s.work[i];
            if (!w->completed) continue;
            if (w->work_type == WORK_FPGA_POP) fp += w->duration;
            else if (w->work_type == WORK_FPGA_NL) fn += w->duration;
            else if (w->work_type == WORK_FPGA_ATTN) fa += w->duration;
        }
        printf("FPGA: pop=%.1f%%, nl=%.1f%%, attn=%.1f%%\n",
               100.0*fp/s.cycle, 100.0*fn/s.cycle, 100.0*fa/s.cycle);
    }

    /* Per-layer timing */
    printf("\n=== Layer Timing ===\n");
    {
        int prev_end = 0;
        int layer_start = 0;
        int projs_per_layer = 7;
        for (int i = 0; i < n_milestones; i++) {
            int wid = milestones[i].wid;
            WorkItem *w = &s.work[wid];
            if (!w->completed) continue;
            int dur = w->end_cycle - prev_end;
            int proj_in_layer = i % projs_per_layer;

            if (i < projs_per_layer) {
                printf("  L1 %-8s  %8d tCK = %6.2f ms\n",
                       milestones[i].name, dur, dur * tck_ns / 1e6);
            }

            if (proj_in_layer == 0) layer_start = prev_end;

            if (proj_in_layer == projs_per_layer - 1 || i == n_milestones - 1) {
                int layer_dur = w->end_cycle - layer_start;
                if (i < projs_per_layer) {
                    printf("  L1 TOTAL:    %8d tCK = %6.2f ms\n",
                           layer_dur, layer_dur * tck_ns / 1e6);
                } else if (i == n_milestones - 1 && strcmp(milestones[i].name, "unembed") == 0) {
                    printf("  Unembed:     %8d tCK = %6.2f ms\n",
                           dur, dur * tck_ns / 1e6);
                }
            }
            prev_end = w->end_cycle;
        }
        if (n_milestones > projs_per_layer) {
            int last_layer_wid = milestones[n_milestones - 2].wid;
            int layers_end = s.work[last_layer_wid].end_cycle;
            double avg_layer = (double)layers_end / n_layers_run * tck_ns / 1e6;
            printf("  Avg layer:   %.2f ms x %d = %.2f ms\n",
                   avg_layer, n_layers_run, avg_layer * n_layers_run);
        }
    }

    /* End-to-end estimate */
    printf("\n=== End-to-End Estimate ===\n");
    printf("Prompt tokens: %d\n", seq_len);
    printf("Prefill: ~%d x %.2f ms = %.1f ms\n",
           seq_len, total_ms, seq_len * total_ms);
    printf("First token latency: %.1f ms\n", (seq_len + 1) * total_ms);
    printf("Decode throughput: %.2f tok/s\n", tok_s);

    /* Incomplete items */
    if (s.n_completed < s.n_work) {
        printf("\nINCOMPLETE ITEMS (first 30):\n");
        int shown = 0;
        for (int i = 0; i < s.n_work && shown < 30; i++) {
            WorkItem *w = &s.work[i];
            if (!w->completed) {
                printf("  wid=%d type=%s dimm=%d bank=%d deps=%d started=%d\n",
                       i, work_type_name[w->work_type], w->dimm, w->bank,
                       w->n_remaining_deps, w->started);
                shown++;
            }
        }
    }

    sched_free(&s);
    return 0;
}
