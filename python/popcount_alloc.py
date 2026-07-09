#!/usr/bin/env python3
# In-place MAJ3-DAG allocator (ADR-004), greedy + spill. Embeds a MAJ3 dataflow
# graph onto co-activatable 4-row tuples. Each gate-output flows IN PLACE to a
# consumer via a shared (carrier) row when the tuple geometry allows; otherwise
# the input is SPILLED (host read-back + write) into a fresh tuple row. Maximizes
# in-place hand-offs; spills bridge interlock-density gaps (the SIMDRAM/MVDRAM
# RowClone-style operand move). Row reuse via liveness (a node's rows free once
# all its consumers are placed). Emits a plan for mvdram-plan-exe; reports the
# in-place vs spill ratio.
import csv, sys, ast, collections, os

S_ID=int(os.environ.get('ALLOC_SID','86'))
OPEN=os.environ.get('ALLOC_OPEN','/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/FindOpenRows/dimm2/open_rows_50.csv')
COV=os.environ.get('ALLOC_COV','/home/deni/Claude/SiMRA-DRAM-main/experimental_data/simra_full_2026_05_20/dimm2/maj_coverage_4_50.csv')
FS_MIN=float(os.environ.get('ALLOC_FS','80'))
WHICH=sys.argv[1] if len(sys.argv)>1 else 'pc4'
OUT=sys.argv[2] if len(sys.argv)>2 else f'/home/deni/Claude/mvdram-repro/plan_{WHICH}.txt'

rows4={}
for r in csv.reader(open(OPEN)):
    if len(r)<12 or r[0]=='': continue
    try:
        if int(r[9])!=S_ID or int(r[3])!=4 or int(r[10])!=0 or int(r[11])!=0: continue
        rows4[(int(r[1]),int(r[2]))]=tuple(ast.literal_eval(r[6]))
    except: continue
grade={}
for r in csv.reader(open(COV)):
    try:
        if int(r[8])!=S_ID or int(r[9])!=3 or int(r[1])!=0 or int(r[2])!=0: continue
        k=(int(r[6]),int(r[7])); grade[k]=max(grade.get(k,0),float(r[13]))
    except: continue
TUP=[(k[0],k[1],rows4[k],grade.get(k,0)) for k in rows4 if grade.get(k,0)>=FS_MIN]
TUP.sort(key=lambda t:-t[3])
print(f"tuple pool (MAJ3-grade fs>={FS_MIN}, s{S_ID}, t12=t23=0): {len(TUP)}", file=sys.stderr)

LITS={'a','b','c','d','na','nb','nc','nd','Z','O'}
DAGS={
 'pc2':([('e1',['a','nb','Z']),('e2',['na','b','Z']),('bit0',['e1','e2','O']),
         ('bit1',['a','b','Z'])],
        [('bit0',0),('bit1',1)]),
 'pc4':([('m1',['a','b','nc']),('m2',['a','nb','c']),('m3',['na','b','c']),
         ('carry1',['a','b','c']),('sum1',['m1','m2','m3']),
         ('m4',['na','b','nc']),('m5',['na','nb','c']),('nsum1',['m4','m5','carry1']),
         ('carry2',['sum1','d','Z']),('ncarry1',['na','nb','nc']),('ncarry2',['nsum1','nd','O']),
         ('p1',['sum1','nd','Z']),('p2',['nsum1','d','Z']),('bit0',['p1','p2','O']),
         ('q1',['carry1','ncarry2','Z']),('q2',['ncarry1','carry2','Z']),('bit1',['q1','q2','O']),
         ('bit2',['carry1','carry2','Z'])],
        [('bit0',0),('bit1',1),('bit2',2)]),
}
GATES,OUTS=DAGS[WHICH]

# consumer counts for liveness (later gate-input uses + pinned if OUT)
out_nodes={n for n,_ in OUTS}
total_cons=collections.defaultdict(int)
for _,ins in GATES:
    for x in ins:
        if x not in LITS: total_cons[x]+=1
PIN=10**9
remaining={n:(total_cons[n]+(PIN if n in out_nodes else 0)) for n,_ in GATES}

occupied={}       # row -> node currently stored
def is_free(r): return (r not in occupied) or (remaining.get(occupied[r],0)<=0)

plan=[]; n_inplace=0; n_spill=0
for gi,(node,ins) in enumerate(GATES):
    gin=[x for x in ins if x not in LITS]; lin=[x for x in ins if x in LITS]
    best=None
    for ti,(rf,rs,rows,fs) in enumerate(TUP):
        rowset=set(rows)
        # in-place carriers: distinct rows occupied by a (live) gate-input
        cmap={}; usedc=set()
        for g in gin:
            for r in rows:
                if r in usedc: continue
                if occupied.get(r)==g and remaining.get(g,0)>0:
                    cmap[g]=r; usedc.add(r); break
        carriers=set(cmap.values())
        noncar=[r for r in rows if r not in carriers]
        if any(not is_free(r) for r in noncar): continue   # a needed row is occupied by live data
        spilled=[g for g in gin if g not in cmap]
        need=len(spilled)+len(lin)+1
        if len(noncar)<need: continue
        score=len(cmap)
        if best is None or score>best[0]: best=(score,ti,cmap,spilled,noncar)
    if best is None:
        print(f"NO TUPLE for gate {node} (all tuples have live-occupied rows) — increase pool/reuse", file=sys.stderr); sys.exit(1)
    score,ti,cmap,spilled,noncar=best
    rf,rs,rows,fs=TUP[ti]
    rr=list(noncar)
    spill_rows=dict(zip(spilled,rr[:len(spilled)])); off=len(spilled)
    lit_rows=dict(zip(lin,rr[off:off+len(lin)])); off+=len(lin)
    frac=rr[off]
    # emit
    for g,r in spill_rows.items(): plan.append(f"SPILL {r} {g}"); n_spill+=1
    for lit,r in lit_rows.items(): plan.append(f"WRITE {r} {lit}")
    plan.append(f"WRITE {frac} O")
    out_row=[r for r in rows if r!=frac][0]
    plan.append(f"MAJ {rf} {rs} {frac} {out_row} {node} {ins[0]} {ins[1]} {ins[2]}")
    n_inplace+=len(cmap)
    # update liveness: consumers of carried inputs are this gate; decrement ALL gate-inputs' remaining
    for g in gin: remaining[g]-=1
    # this gate now occupies all its tuple rows
    for r in rows: occupied[r]=node
for n,b in OUTS: plan.append(f"OUT {n} {b}")
open(OUT,'w').write('\n'.join(plan)+'\n')
tot=n_inplace+n_spill
print(f"ALLOCATED {WHICH}: {len(GATES)} gates -> {OUT}", file=sys.stderr)
print(f"  gate-input hand-offs: {n_inplace} in-place + {n_spill} spill = {tot}  ({100.0*n_inplace/max(tot,1):.0f}% in-place)", file=sys.stderr)
