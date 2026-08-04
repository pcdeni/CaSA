# Public-docs restructure — what was kept, cut, merged, and why

*2026-08-04. This note records the structural decisions so the next pass does
not re-litigate them. The prior passes grew the surface by addition — seven
explainer decks, two comparison homes. This pass evaluated each page against the
reader's jobs and collapsed the surface to one home per job.*

## The reader's jobs (a layman-facing LLM-in-DRAM repo)

1. **Front page** — what this is, in one screen.
2. **One walkthrough** — how an LLM runs in DRAM, end to end.
3. **One mechanism page** — the physics.
4. **Comparison + how we measure** — peers, the wall model, verification.
5. **Roadmap** — open levers and status.

Each surviving page must read top to bottom for someone with no DRAM
background, in the **present tense** (no build numbers, no belief-history, no
"we used to think"), and must state each fact **once** — every other page links
to that home rather than restating it.

## Decision table

| Page | Job | Decision | Why |
|---|---|---|---|
| `index.html` — *How an LLM runs inside a DRAM chip* | 2 walkthrough | **KEEP** (the home) | The 9-scene plain-language tour, cell → inference loop. Gestures at the mechanism once and links out. |
| `xor-spread.html` — *One command pair, two physics* | 3 mechanism | **KEEP** (the home) | The one place the family rule + timing dial + copy/vote spread is taught. Opens with the three-fact present-state statement. |
| `docs/RELATED_SYSTEMS.md` — *Related systems & methodology* | 4 comparison | **KEEP** (the home) | Owns the MVDRAM map (§2), the rigor/verification rules (§4), the wall model (§5), the measurement caveats (§6). Both explainers and the README already point here as the single home for those. |
| `docs/ROADMAP.md` | 5 roadmap | **KEEP** (the home) | Present-state mirror of the living levers ledger. |
| `README.md` (root) | 1 front page | **KEEP** (the home) | One-screen thesis + ladder + seven models + repo map; its "where to go next" table names only the four homes above. |
| `system.html` | — | **CUT → deleted → `index.html`** | Its job *is* the walkthrough; it overlapped `index.html` scene for scene. |
| `dram-internals.html` | — | **CUT → deleted → `xor-spread.html`** | Substrate basics are covered in `index.html` scenes 2–3; the atlas depth belongs with the mechanism. |
| `optimization-spine.html` | — | **CUT → deleted → `index.html`** | A diary by construction ("the chain of improvements"). The live throughput numbers live in the README and RELATED_SYSTEMS §5; status lives in ROADMAP. |
| `experiment-trail.html` | — | **CUT → deleted → `index.html`** | A diary by construction ("what was tried"). Belief-history has no place on the layman surface. |
| `mvdram.html` | 4 (dup) | **CUT → deleted → RELATED_SYSTEMS §2 + MVDRAM_REPRODUCTION.md** | It was a *second* comparison home and re-taught the mechanism in diary form. Comparison is job 4's home (RELATED_SYSTEMS); the deep reproduction study is `MVDRAM_REPRODUCTION.md`. |

## What moved where

- **MVDRAM comparison** → already in `RELATED_SYSTEMS.md` §2. The deck's one
  genuinely unique fact — the **PuDGhost** independent corroboration
  (arXiv:2606.19119, the MVDRAM authors' own group) — was folded into §2.5, the
  error-model section, present-tense and cited.
- **Circular-link fix:** `RELATED_SYSTEMS.md` §2 previously linked the deck;
  since the deck now redirects *to* RELATED_SYSTEMS, that link was repointed to
  `MVDRAM_REPRODUCTION.md` (the deep study).
- **`index.html` "go deeper" tiles** shrank from nine (four stubs + the mvdram
  deck + the four homes) to the **four surviving homes**: the mechanism, the
  comparison/methodology doc, the roadmap, the front page.
- **`xor-spread.html`** now opens with the three-fact present-state statement —
  the permanent decoder-wired address family (fixed XOR offsets), content
  spreading across that family at **copy** timing (a real, measured RowClone,
  the reason weights are placed to avoid collisions), and nothing imported at
  **vote** timing (the rows are counted, majority wins). Belief-history and
  narrative past tense were removed.

## Ledgers

The two current publish gates stay live and accurate:
`xor-spread_ledger_2026_08_03.md` (mechanism) and `index_ledger_2026_08_03.md`
(system). Three superseded ledgers were marked **RETIRED** with a top banner and
a pointer to their successor: `xor_spread_ledger.md`, `pim_explainer_ledger.md`,
and `mvdram_explainer_ledger.md`. `paper_mechanism_notes.md`,
`pim_explainer_review.md`, and `publish_ledger_2026_07_20.md` remain as the
supporting evidence trail.

## Docs-tier stubs (unchanged pattern, listed for completeness)

`METHODOLOGY.md`, `UTILIZATION.md`, `MVDRAM_COMPARISON.md`, `PAPER_CONTRAST.md`,
and `LATTICE_ADDRESSING_2026_07.md` are already one-line redirects into
`RELATED_SYSTEMS.md` / the mechanism explainer — the same stub pattern used for
the cut decks.


Cut pages are deleted, not stubbed: the repo is young, no external
deep links depend on those URLs, and a "this page has moved" tombstone
is clutter a reader can stumble into. Git history preserves everything.
