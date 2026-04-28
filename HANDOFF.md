# Handoff Document: 120 Composed 2D GEMM Algorithms

## Goal
Complete a configuration-driven library of 120 composed 2D GEMM algorithms in `composed_gemm.py`. These are all combinations of 2 1D algorithms (outer × inner loop), covering 15 unordered pairs × 2 orderings × 4 ring-direction combos (PREV/NEXT × PREV/NEXT).

The system is already built — `Gemm1D`, `Gemm2D`, `GEMM_2D_INNER_CONFIGS`, and `GEMM_2D_ALGORITHMS` all exist. Only data entries are missing, not engine code.

---

## Current State

| What | Done | Total |
|------|------|-------|
| Forward GEMM_2D_INNER_CONFIGS | 15 | 15 |
| Forward GEMM_2D_ALGORITHMS | 15 | 15 |
| Reversed GEMM_2D_INNER_CONFIGS | 4 | 15 |
| Reversed GEMM_2D_ALGORITHMS | 4 | 15 |
| Composed tests in tests.py | 76 | 120 |

**76 tests pass** (confirmed baseline). 44 tests missing = 11 reversed pairs × 4 directions.

---

## The 11 Missing Reversed Pairs

Each is the swap of a forward algorithm's (outer, inner):

| Rev of | Forward key | **Missing key** |
|--------|-------------|-----------------|
| Alg 2  | (AG_A_COL, AG_B_COL) | **(AG_B_COL, AG_A_COL)** |
| Alg 3  | (AG_B_ROW, AG_A_COL) | **(AG_A_COL, AG_B_ROW)** |
| Alg 4  | (RS_C_COL, AG_A_COL) | **(AG_A_COL, RS_C_COL)** |
| Alg 5  | (RS_C_ROW, AG_A_COL) | **(AG_A_COL, RS_C_ROW)** |
| Alg 7  | (AG_B_ROW, AG_A_ROW) | **(AG_A_ROW, AG_B_ROW)** |
| Alg 8  | (AG_A_ROW, RS_C_COL) | **(RS_C_COL, AG_A_ROW)** |
| Alg 9  | (RS_C_ROW, AG_A_ROW) | **(AG_A_ROW, RS_C_ROW)** |
| Alg 11 | (RS_C_COL, AG_B_COL) | **(AG_B_COL, RS_C_COL)** |
| Alg 12 | (AG_B_COL, RS_C_ROW) | **(RS_C_ROW, AG_B_COL)** |
| Alg 13 | (RS_C_COL, AG_B_ROW) | **(AG_B_ROW, RS_C_COL)** |
| Alg 14 | (RS_C_ROW, AG_B_ROW) | **(AG_B_ROW, RS_C_ROW)** |

---

## Files to Modify

- `composed_gemm.py`: Add 11 entries after line 603 in `GEMM_2D_INNER_CONFIGS`, 11 entries after line 938 in `GEMM_2D_ALGORITHMS`
- `tests.py`: Add 44 entries (11 pairs × 4 directions) after line 290 in `GEMM_TESTING_CONFIGURATIONS`

---

## How to Derive Each Reversed Algorithm

**Rule 1**: `group_param_is_py` always flips relative to the forward algorithm.

**Rule 2**: For communicators:
- `nearby_rank_communicator(comm, group_param, rank)` → splits by `rank // group_param` → ic = `rank % group_param`, inner size = `group_param`
- `remainder_communicator(comm, group_param, rank)` → splits by `rank % group_param` → oc = `rank // group_param`, outer size = `size / group_param`

**Rule 3**: Distributions must be rederived. The safest approach is to look at the thesis (CommHidingThesis.pdf) for each reversed algorithm's distribution table, OR derive by analogy:
- `row_major(px, py, rank)` = `get_subtile(M, px, py, rank//py, rank%py)` = M[oc_when_gp=py, ic_when_gp=py]
- `row_major(py, px, rank)` = M[oc_when_gp=px, ic_when_gp=px]
- `col_major(px, py, rank)` = `get_subtile(M, px, py, rank%px, rank//px)`

**The 4 existing reversed algorithms as templates:**
- R1 (rev Alg1: AG_A_COL→AG_A_ROW): gp flips True→False; A dist: `row_major(px,py)` → `row_major(py,px)`. Same B/C.
- R10 (rev Alg10: AG_B_COL→AG_B_ROW): gp flips False→True; B dist: `col_major(px,py)` → `row_major(px,py)`. Same A/C.
- R15 (rev Alg15: RS_C_COL→RS_C_ROW): gp flips False→True; C dist: `col_major(px,py)` → `col_major(py,px)`. Same A/B.
- R6 (rev Alg6: AG_A_ROW→AG_B_COL): gp flips True→False; B: `pure_col` → `alternating_col(py,px)`; C: `col_major(py,px)` → `row_major(py,px)`.

**Inner config derivation**: The inner config for `(outer=Y, inner=X)` reversed typically mirrors the inner config of whichever forward algorithm has the same inner matrix type `X`. Key differences:
- The buffer dimensions swap `os_` and `is_` roles
- The `get_subtile` indices in `tiles` and `set_c` change to reflect the swapped oi/sz roles
- `persistent_buffer` and `loopback` flags follow the outer matrix type (AG outer → typically no persistent buf; RS outer → varies)

---

## Inner Config Reference (what each lambda arg means)

In `GEMM_2D_INNER_CONFIGS`:
- `oi` = outer index (which outer iteration we're on, 0..outer_size-1)
- `os_` = outer size (total outer iterations)
- `is_` = inner size (total inner iterations)
- `idx` = inner index (current inner ring step, 0..sz-1)
- `sz` = inner communicator size
- `buf` = the rotating inner buffer (DoubleBuffer or SubtileBuffer)

In `tiles`:
- `A_curr`, `B_curr`, `C_curr` are lambdas `(A, B, C, buf, idx, sz) -> tile`
- `C_curr` for AG outer: slice of C local tile (what this rank accumulates into)
- `C_curr` for RS outer: usually `np.zeros(C.shape)` (fresh accumulator)

---

## Test Entry Pattern

For each reversed pair, add 4 entries to `GEMM_TESTING_CONFIGURATIONS` in `tests.py`:

```python
# RX: outer=Y, inner=X — min dims: m%?, k%?, n%?
"Y_PREV_X_PREV_COMPOSED": TestGemmConfiguration(
    Gemm2D(Gemm1D(MatrixCommunicated.Y_MAT, SubtileScheme.Y_SCHEME, CommunicationDirection.SEND_PREV),
           Gemm1D(MatrixCommunicated.X_MAT, SubtileScheme.X_SCHEME, CommunicationDirection.SEND_PREV)).setup_and_run,
    GemmDimension.M_DIV, GemmDimension.K_DIV, GemmDimension.N_DIV),