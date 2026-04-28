# 2D Composed GEMM: In-Depth Codebase Guide

This document covers only the 2D composition machinery in `composed_gemm.py`. It explains not just what each piece does but why it had to be designed the way it was. Assumes familiarity with the 1D ring loop in `Gemm1D.run()` and basic MPI concepts.

---

## Table of Contents

1. [The Core Problem: Why 2D Is Not Just "Run 1D Twice"](#1-the-core-problem)
2. [The Two Sub-Communicators](#2-the-two-sub-communicators)
3. [GEMM_2D_OUTER_CURRENT_TILES and _noop_set_c](#3-outer-current-tiles-and-noop)
4. [GEMM_2D_INNER_CONFIGS: Every Field, Every Reason](#4-inner-configs)
5. [_make_compute_fn: The Composition Engine](#5-make-compute-fn)
6. [GEMM_2D_ALGORITHMS: Distribution Engineering](#6-algorithms-dict)
7. [Gemm2D: Wiring It All Together](#7-gemm2d)
8. [Deep Dives: Four Representative Algorithms](#8-deep-dives)

---

## 1. The Core Problem

A 2D GEMM distributes `C = A×B + C` across `P = px × py` processes. A 1D algorithm only covers `P` processes as a single ring. To compose two 1D algorithms into a 2D one, you nest them:

```
for each outer_step in outer_ring (size = px or py):
    for each inner_step in inner_ring (size = py or px):
        local_result = matmul(A_curr, B_curr)
        accumulate into C_curr
```

The outer loop does `outer_size` steps over one sub-communicator; the inner loop does `inner_size` steps over a different sub-communicator. Together they cover all `P` rank-pairs.

**Why you can't just call two 1D algorithms sequentially:** A 1D algorithm is self-contained — it distributes matrices, runs a ring, gathers results, and checks correctness. It has no concept of "I'm the outer loop and my current A-tile should be passed to the inner loop." The 2D design requires:

1. The outer loop does NOT write C itself. Instead, at each outer step, it calls a `compute_fn` that runs the entire inner loop with the current outer context (the current A or B tile).
2. The inner loop uses a different C subtile at each outer step, addressed by the outer step index `oi`.
3. Some inner buffers need to persist across outer steps (not reset each time) because they are rotating continuously through the combined outer+inner ring.
4. When the inner loop is ReduceScatter, its accumulated result needs special handling after the inner loop finishes — it must be merged into the real C in a way the outer loop doesn't know about.

The entire 2D machinery exists to solve these four problems.

---

## 2. The Two Sub-Communicators

With `P = px * py` processes, you need two independent rings. `communicator.py` provides two splitting strategies:

```python
nearby_rank_communicator(comm, group_size, rank):
    return comm.Split(rank // group_size, rank)
# Ranks {0,1,2}, {3,4,5}, {6,7,8}, ... form groups

remainder_communicator(comm, num_groups, rank):
    return comm.Split(rank % num_groups, rank)
# Ranks {0,3,6}, {1,4,7}, {2,5,8}, ... form groups
```

For `P=6, px=2, py=3`: `nearby_rank_communicator(3, rank)` creates groups `{0,1,2}` and `{3,4,5}`. `remainder_communicator(3, rank)` creates groups `{0,3}`, `{1,4}`, `{2,5}`.

**Why these two specific splits?** The two splits are complementary: every rank is in exactly one group in each split, and the two groupings partition the process grid along orthogonal dimensions. When you run an outer ring over one split and an inner ring over the other, every pair of ranks (r_outer, r_inner) is covered exactly once across all outer+inner steps. This is the mathematical condition required for the 2D algorithm to compute every partial product exactly once.

**Why AG-outer uses remainder and RS-outer uses nearby (line 1321–1326):**

This is a convention about how the process grid maps to communicators. For the 15 forward algorithms, the "outer communicator" is the one associated with the outer matrix's distribution. For example, Alg 1 has outer=AG_A_COL (A distributed column-wise, outer comm = py-sized groups). The `remainder_communicator(py, rank)` groups ranks with the same `rank % py`, which corresponds to ranks in the same column of the `px × py` grid. This aligns with a column-based distribution.

The reversed algorithms (R1–R15) swap which communicator is outer/inner, which is precisely what makes them "reversed" — the same distributions are applied but the ring traversal order is different.

**`group_param_is_py` in `GEMM_2D_ALGORITHMS`:** This flag says whether to use `py` (True) or `px` (False) as the group size when splitting. It controls how many processes are in each ring. For example, Alg 1 has `group_param_is_py=True`, so the outer ring has `py` processes and the inner ring has `px` processes. Alg 2 has `group_param_is_py=False`, so outer ring = `px` processes, inner ring = `py` processes. This determines the shape of the sub-problem each ring handles.

---

## 3. GEMM_2D_OUTER_CURRENT_TILES and `_noop_set_c`

```python
GEMM_2D_OUTER_CURRENT_TILES = {
    MatrixCommunicated.A: CurrentTiles(
        A_curr=lambda A, B, C, buffer, index, size: buffer.get_buffer(),
        B_curr=lambda A, B, C, buffer, index, size: B,
        C_curr=lambda A, B, C, buffer, index, size: np.zeros(shape=C.shape, dtype=MATRIX_DTYPE)
    ),
    # ... B and C variants are similar
}
```

**Why this exists:** In a standalone 1D algorithm, `current_tiles` is responsible for slicing the appropriate subtile of each matrix at each ring step. But in the 2D context, the outer loop must NOT do any subtiling itself — that's the inner loop's job. The outer loop just needs to rotate its matrix (A or B) and expose the current rotated tile to `compute_fn`. Everything else (subtile addressing, C writes) is delegated downward.

**Why `C_curr` is always `np.zeros`:** The outer loop computes `C_tmp = compute_fn(A_outer, B_outer, i, index) + C_curr`. Since the inner loop is responsible for writing C, `compute_fn` returns `np.zeros(C.shape)` (a dummy zero). Adding `C_curr = zeros` to that gives zero, and then `set_c` (which is `_noop_set_c`) throws it away. The outer loop's C machinery is completely bypassed. The zeros are necessary syntactically because `Gemm1D.run()` always computes `C_tmp = result + C_curr` regardless.

**Why `_noop_set_c` instead of just never calling set_c:** `Gemm1D.run()` always calls `set_c(C, C_tmp, index, size)` for non-RS loops. You can't skip it structurally. The no-op override is a clean way to disable C writes from the outer loop without changing `Gemm1D.run()`. This keeps `Gemm1D` single-purpose: it runs a ring, calls whatever set_c it's given. In the 2D case it's given a no-op.

**The exception — outer RS:** For outer RS (C communicated), `Gemm1D.run()` takes a different code path for the matrix_communicated==C branch (lines 248–253): it accumulates C_tmp into a ring and does `C = C + C_tmp` at the last step. The `C_tmp` here is `compute_fn(A_outer, B_outer, i, index) + zeros`. So the outer RS ring accumulates the return values of `compute_fn` across outer steps. This is fundamentally different from AG outer — when outer is RS, `compute_fn` must return a meaningful partial sum, not zeros. More on this in section 4.

---

## 4. GEMM_2D_INNER_CONFIGS: Every Field, Every Reason

`GEMM_2D_INNER_CONFIGS` is the heart of the composition. It is a 30-entry dict keyed by `((outer_matrix, outer_subtile), (inner_matrix, inner_subtile))`. Each entry answers: **"given that the outer loop is at step `oi` with tile context `(A_outer, B_outer)`, how should the inner loop be configured?"**

### 4.1 `make_buffer`

**Signature:** `lambda A, B, C, A_outer, B_outer, oi, is_, os_ → Buffer`

Where `A`, `B`, `C` are the full local matrices (owned by this rank); `A_outer`, `B_outer` are the outer loop's current rotated tiles; `oi` is the outer step index; `is_` is inner ring size; `os_` is outer ring size.

**Why `make_buffer` receives both the full local matrices AND the outer context:** Different algorithms need different things to initialize the inner buffer:

- **Alg 1** (`outer=AG_A_COL, inner=AG_A_ROW`): Inner loop rotates A. The starting tile is `A_outer` (the current outer A-column-strip). So `make_buffer` uses `np.copy(A_outer)`. The copy is necessary because `A_outer` is a view into the outer loop's `DoubleBuffer`, which will swap to a different value when the outer loop advances. The inner loop must own its A independently.

- **Alg 2** (`outer=AG_A_COL, inner=AG_B_COL`): Inner loop rotates a subtile of B. The starting tile is column `oi` of B (from the full `B` matrix). So `make_buffer` uses `SubtileBuffer(B, os_, 1, oi, 0)` — a buffer wrapping column `oi` of B. `oi` is used here, so the buffer changes each outer step, which is why `persistent_buffer=False`.

- **Alg 6** (`outer=AG_A_ROW, inner=AG_B_COL`): Inner loop rotates all of B through a persistent ring. The buffer wraps the entire `B` matrix. `A_outer` is not used — the inner loop gets `A_outer` from the outer context via `tiles`, not via the buffer.

**Why `make_buffer` returns `None` for RS inner loops:** When the inner loop is RS_C, the inner loop uses an `AccumulationBuffer` that is instantiated by the 1D config machinery (`GEMM_1D_CONFIGURATIONS[BUFFER][C]`), not by the 2D config. The 2D config doesn't override the buffer for RS inner. A `None` `make_buffer` signals "_let Gemm1D.run() create its own buffer from the 1D config_" — which the code handles at line 808: `buffer = None` gets passed as `buffer_override=None` to `Gemm1D.run()`, causing it to fall back to `self.config.buffer(A, B, C)`.

---

### 4.2 `persistent_buffer`

**What it does:** If `True`, `make_buffer` is called once in the factory scope (before any outer step begins) and the same buffer object is reused across all `outer_size` inner loop runs.

**Why this is necessary for correctness (not just performance):**

Consider `DoubleBuffer`. It wraps two numpy arrays and alternates between them. Each call to `on_receive()` triggers a `swap()`, so after `N` swaps, `current_buffer` points to `first_buffer` if N is even, `second_buffer` if N is odd.

In a `loopback=True` inner loop of size `inner_size`, there are `inner_size` swaps total (one per receive, including the loopback receive). The data — the A or B tile — travels a full ring and returns to the starting rank's memory. BUT `current_buffer` now points to `first_buffer` if `inner_size` is even, `second_buffer` if odd.

**If `inner_size` is odd and `persistent_buffer=False`:**  
The buffer is recreated at the start of each outer step: `DoubleBuffer(A)` assigns `first_buffer = A` (a reference to the numpy array). After an odd number of swaps, `current_buffer = second_buffer`, and `first_buffer = A` still points to the original array. So far so good — the data traveled a full ring and returned to `A`. The next outer step creates a fresh buffer pointing to `A`, and the cycle repeats. **This is actually correct.**

So why is `persistent_buffer` needed at all? The answer is more subtle: for **Alg 4 and 6**, the inner buffer must survive across outer steps because its content is being modified by the outer ring too. In Alg 4 (outer=RS_C_COL, inner=AG_A_COL), A rotates in the inner ring with loopback, but the outer ring is simultaneously rotating C. The A buffer must maintain continuity — the buffer state at the end of one outer step is the correct starting state for the next outer step, because the outer loop's ring step has moved A to a different rank in the outer ring. Recreating the buffer from `A` (the local numpy array) each outer step would ignore the outer ring's effect on where A currently lives.

**In short:** `persistent_buffer=True` is required when the buffer represents a matrix that is being moved by the outer ring at the same time the inner loop is using it. The buffer tracks "current" identity of the tile across both loops simultaneously.

**Why persistent buffers are created with `(A, B, C, None, None, None, inner_size, outer_size)`:** The `A_outer`, `B_outer`, `oi` arguments are `None` because at factory-creation time (before the outer loop starts), there is no outer context yet. The buffer lambdas that use `persistent_buffer=True` are written to use only `A`, `B`, `C`, `is_`, `os_` — never `A_outer`, `B_outer`, or `oi`. This is enforced by convention: if a buffer lambda uses `oi`, it can't be persistent because `oi` changes each step.

---

### 4.3 `loopback`

**What it is:** If `True`, `should_send` is True on the last inner step too. This causes `inner_size` sends instead of `inner_size - 1`.

**Why it exists:** In a standard AG ring of size N, you do N-1 sends. Each rank starts with one tile, and after N-1 receives, it has seen all N tiles. The last step has nothing left to send forward. In 1D this is fine because the algorithm is done and each rank keeps its final received tile.

In the 2D context with a persistent buffer, the inner loop's final state (what tile the buffer is holding) carries over to the next outer step. After `inner_size - 1` sends, the buffer holds the tile from rank `(start - inner_size + 1) mod inner_size` — **not** the tile it started with. If the outer loop then advances (rotating A or B to a new outer position), the inner buffer is holding the wrong tile for the next inner loop run.

**With `loopback=True`:** After `inner_size` sends, the buffer holds the tile that was sent in step 0, which has traveled all the way around the ring and come back. The buffer is back to its starting content. The next outer step can start fresh with the same initial tile.

**Why only some algorithms need loopback:** Algorithms with `persistent_buffer=True` need `loopback=True`. Non-persistent algorithms recreate the buffer each outer step from scratch, so there's no carried-over state to fix. Some non-persistent algorithms also use `loopback=True` (Alg 11, 13): these use a `SubtileBuffer` where the send/receive writes back into the parent matrix, and the loopback is needed to return that subtile to its original rank so the parent matrix is restored.

**Why loopback is disabled on the last outer step (`loopback = config['loopback'] and not is_last_outer`):** After the last outer step, there are no more inner loops to run, so there's no reason to pay for the extra communication. The buffer state at the end of the last outer step doesn't matter.

---

### 4.4 `tiles`

**Signature:** `lambda A_outer, B_outer, oi, os_ → CurrentTiles`

Returns a `CurrentTiles` object whose three lambdas (`A_curr`, `B_curr`, `C_curr`) tell the inner loop what tiles to use at each inner step.

**Why it's a lambda that produces a CurrentTiles (not just a CurrentTiles directly):** The `tiles` entry is called at the start of each outer step with the current `(A_outer, B_outer, oi, os_)`. The inner lambdas then close over these values. If `tiles` were just a static `CurrentTiles`, it couldn't incorporate `oi` (the outer step index) into the subtile addresses for C. By making it a factory, the outer context is baked in each time.

**How to read the inner lambdas inside `tiles`:**

The parameters are `(A, B, C, buf, idx, sz)` where:
- `A`, `B`, `C` are the local full matrices passed through from `Gemm1D.run()`
- `buf` is the inner loop's buffer (from `buffer_override`)
- `idx` is the current inner step index
- `sz` is the inner ring size

The closed-over values `A_outer`, `B_outer`, `oi`, `os_` come from the outer loop's context.

**Example — Alg 1** (`outer=AG_A_COL, inner=AG_A_ROW`):

```python
'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
    A_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
    B_curr=lambda A, B, C, buf, idx, sz: get_subtile(B, sz, 1, idx, 0),
    C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, os_, 1, oi, 0),
),
```

- `A_curr`: The inner loop is AG_A_ROW (rotating A). A_curr is the current A tile from the inner buffer.
- `B_curr`: B is not communicated by the inner loop. It's split into `sz` row-bands; `idx` selects which row-band to use at inner step `idx`. This is how each inner step multiplies by a different K-block of B.
- `C_curr`: C is sliced into `os_` row-bands (one per outer step). `oi` selects the band for this outer step. `idx` is not used here because the full C subtile for outer step `oi` is used at every inner step (inner AG accumulates into it).

**Example — Alg 2** (`outer=AG_A_COL, inner=AG_B_COL`):

```python
'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
    A_curr=lambda A, B, C, buf, idx, sz: A_outer,
    B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
    C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, 1, sz, 0, idx),
),
```

- `A_curr`: The outer loop gave us this column-strip of A. The inner loop doesn't communicate A at all — it just uses `A_outer` at every inner step.
- `B_curr`: The inner loop rotates B (AG_B_COL). `buf.get_buffer()` gives the current rotated column-strip of B.
- `C_curr`: C is split into `sz` column-bands (one per inner step index). `idx` selects which C column-band to update at each inner step. `oi` is not used because AG_B_COL splits C by inner index, not outer index.

The key pattern: whichever matrix the inner loop communicates, `C_curr` addresses the output subtile using the index of the OTHER loop. Specifically:
- Inner communicates A → C is indexed by `oi` (outer step selects the output row-band)
- Inner communicates B → C is indexed by `idx` (inner step selects the output column-band)
- Inner communicates C → C_curr is zeros; C accumulation happens differently

---

### 4.5 `set_c`

**Signature:** `lambda A_outer, B_outer, oi, os_ → (lambda C, C_tmp, idx, sz: ...)`

A double-lambda: the outer call captures the outer context; the inner call is the actual write-back function used by `Gemm1D.run()` after each inner matmul.

**Why it mirrors `tiles`:** `set_c` writes the result `C_tmp` back into the same subtile that `C_curr` reads from. The addressing must match exactly. Both use the same `oi` and `os_` to locate the C subtile.

**Why it's `None` for RS inner loops:** When the inner loop is RS_C, `Gemm1D.run()` uses its own C accumulation path (the else branch at line 248). It doesn't call `set_c` at all. Passing `None` as `set_c_override` causes `Gemm1D.run()` to fall back to `self.config.set_c`, but since `make_C_inner` creates a separate scratch C and we pass `set_c_override=None`, the inner RS loop uses its built-in behavior.

**`np.copyto(C, Ct)` vs `set_subtile(C, Ct, ...)`:** Some algorithms (like Alg 3, 4, 5) use `np.copyto(C, Ct)` — they overwrite all of C with C_tmp. This is correct when C_curr and C_tmp are the same shape as C (no subtiling). When C is subtiled (a different portion updated each inner step), `set_subtile` is used to write only the correct portion.

---

### 4.6 `make_C_inner`

**Signature:** `lambda C, oi, is_, os_ → np.ndarray`

Returns a fresh zero-filled array to use as C inside the inner loop, instead of the real C.

**Why it exists — two separate reasons:**

**Reason 1 — Inner loop is RS_C:**  
A ReduceScatter ring accumulates partial sums into C as tiles rotate. The ring expects to START with C = 0, not with whatever partial sum is already in C. If you pass the real C (already containing some accumulated value) into an RS inner loop, every step will incorrectly add to that pre-existing value. `make_C_inner` gives the RS loop a clean zero slate.

**Reason 2 — Outer loop is RS_C:**  
When the outer loop is ReduceScatter, it accumulates the `compute_fn` return values via its own ring. Each outer step, `compute_fn` runs the inner loop and returns a partial sum for this outer step. The outer RS ring then routes these partial sums to the correct ranks. For this to work, the inner loop must NOT accumulate into the real C directly — that would mix partial sums from different outer steps. `make_C_inner` ensures each outer step's inner loop writes into a fresh zero array, and the result is returned to the outer RS ring for proper routing.

**The two reasons can combine:** Alg 15 (outer=RS_C_COL, inner=RS_C_ROW) has `make_C_inner` because both are RS. Alg 4 (outer=RS_C_COL, inner=AG_A_COL) has `make_C_inner` because the outer is RS (reason 2).

**When `make_C_inner` is `None`:** The inner loop writes directly into the real C. This happens for AG outer + AG inner combinations where no partial-sum isolation is needed. Each inner step writes to a specific, non-overlapping subtile of C (addressed by `oi` and `idx` together), so there's no risk of overwriting partial sums from other steps.

---

### 4.7 `rs_final`

**Signature:** `lambda result, C, oi, is_, os_ → None`

Called once after the inner loop completes when `make_C_inner` is set. Merges the inner loop's accumulated `result` back into the real `C`.

**Why it only exists when outer is AG and inner is RS:**

When outer is AG and inner is RS, here is what happens:
- Outer AG communicates A or B and calls `compute_fn` at each step
- Inner RS runs and accumulates into a zero C (`make_C_inner` creates the zero array)
- After the inner loop completes, the result is a partial sum for a specific subtile of C
- This partial sum must be added to the real C at the right subtile location

`rs_final` does exactly that — it takes `result` (the inner RS's accumulated output) and adds it to the appropriate position in the real C.

**Why it fires from inside `_make_compute_fn` and not from `Gemm1D.run()`:** `Gemm1D.run()` doesn't know it's being used as an inner loop inside a 2D algorithm. After the inner RS loop finishes, it returns its C (the zero array that has been accumulated into). `_make_compute_fn` receives that return value as `result`. It then has the outer context (`outer_index`, `outer_size`) available via closure, which is what `rs_final` needs to know WHERE in C to write. `Gemm1D.run()` can't do this because it doesn't have outer context.

**When outer is RS (no `rs_final` needed):** When outer is RS, `_make_compute_fn` returns `result` directly to the outer ring. The outer RS ring routes it to the right rank. `rs_final` is not called; the outer RS loop is the one that decides where the final result goes.

---

## 5. `_make_compute_fn`: The Composition Engine

```python
def _make_compute_fn(alg_key, A, B, C, inner_comm, inner_size, inner_rank,
                     outer_size, px, py, inner_config, outer_config):
```

This function is called once when `Gemm2D.run()` starts (before the outer loop begins). It returns a `compute_fn` closure that the outer loop calls at each step.

**Why a factory returning a closure (not a simple function):**

The inner loop needs to know things that are fixed for the lifetime of a `Gemm2D.run()` call but are not the same across outer steps:

- Fixed for the entire run: `A`, `B`, `C`, `inner_comm`, `inner_size`, `inner_rank`, `outer_size`
- Varies per outer step: `A_outer`, `B_outer`, `i`, `outer_index`

A factory is the standard Python pattern for this: capture the fixed values in the factory call, expose the varying values as `compute_fn` parameters. This avoids passing 12 arguments through the outer `Gemm1D.run()` loop, which doesn't know about 2D composition at all.

**The persistent buffer decision point (lines 796–798):**

```python
persistent_buf = None
if config.get('persistent_buffer') and config['make_buffer']:
    persistent_buf = config['make_buffer'](A, B, C, None, None, None, inner_size, outer_size)
```

If `persistent_buffer=True`, the buffer is created HERE, at factory time, before any outer step. The `None` values for `A_outer`, `B_outer`, `oi` are intentional — persistent buffer lambdas don't use the outer context (they can't, since there is no outer step yet). If `persistent_buffer=False`, `persistent_buf` stays `None` and the buffer is created inside `compute_fn` each outer step.

**The `loopback` decision inside `compute_fn` (line 802):**

```python
loopback = config['loopback'] and not is_last_outer
```

`loopback` is disabled on the last outer step. This avoids one extra ring communication that serves no purpose (the persistent buffer won't be used again after the last inner loop). This is purely an optimization — on the last step, the buffer's final state doesn't matter, so the extra communication is wasteful.

**The control flow for `make_C_inner` and `rs_final` (lines 833–840):**

```python
if config['make_C_inner']:
    if outer_is_ag:
        config['rs_final'](result, C, outer_index, inner_size, outer_size)
        return np.zeros(C.shape, dtype=MATRIX_DTYPE)
    else:
        return result
else:
    return np.zeros(C.shape, dtype=MATRIX_DTYPE)
```

Four cases:

| `make_C_inner` | outer type | What happens |
|---|---|---|
| None | AG | Inner wrote directly to C via set_c. Return zeros (outer loop discards it). |
| None | RS | **Cannot happen** — outer RS always needs `make_C_inner` for partial-sum isolation. |
| Set | AG | Inner RS accumulated into scratch C. Call `rs_final` to merge into real C. Return zeros. |
| Set | RS | Inner AG accumulated into scratch C. Return that scratch C as the partial sum to the outer RS ring. |

**Why outer AG + `make_C_inner=None` returns zeros but outer RS + `make_C_inner=None` can't exist:**  
The outer RS ring accumulates `compute_fn` return values via its ring protocol. If inner wrote directly to C (no `make_C_inner`), there's nothing to return — C was updated in-place. But the outer RS loop still calls `compute_fn` and expects to route the return value. Returning zeros here would cause the outer RS loop to accumulate and distribute zeros, corrupting C. This case simply doesn't occur: all outer-RS algorithms use `make_C_inner` precisely to create a separable partial sum.

---

## 6. `GEMM_2D_ALGORITHMS`: Distribution Engineering

Each of the 30 entries in `GEMM_2D_ALGORITHMS` defines how to distribute matrices across `P = px * py` processes so that the 2D ring loops compute the correct result.

### 6.1 `group_param_is_py`

Determines whether the sub-communicator groups have size `py` (True) or `px` (False). This controls how many processes participate in each ring. The outer ring size equals `group_param` (either `py` or `px`); the inner ring size equals the other.

### 6.2 `assert_div`

Divisibility constraints must be satisfiable given the distribution. For example, if A is distributed with `row_major_distribution(M, px, py, rank)`, then A has shape `(m/px, k/py)`, requiring `m % px == 0` and `k % py == 0`. The assertion lambda enforces this before any MPI communication happens.

### 6.3 Distribution lambdas (`A_dist`, `B_dist`, `C_dist`)

**Signature:** `lambda M, px, py, rank, size, outer_comm, inner_comm → local_tile`

These determine how each matrix is split among ranks. The `outer_comm` and `inner_comm` sub-communicators are passed in because some algorithms use `oc.Get_rank()` or `ic.Get_rank()` to determine which subtile to give a rank. The `rank` and `size` are the global rank/size.

**Why distributions are coupled to communicator structure:**

The inner loop's `tiles` lambdas access local matrices using addresses derived from `oi` (outer step index, which is the rank's position in the outer ring) and `idx` (inner step index, the rank's position in the inner ring). For the computation to be correct, each rank's local tile must correspond exactly to its position in both rings.

For example, in Alg 3 (`outer=AG_B_ROW, inner=AG_A_COL`), A is distributed as `block_cyclic_distribution(M, px, py, oc.Get_rank(), ic.Get_rank())`. This uses both the outer ring rank (`oc.Get_rank()`) and inner ring rank (`ic.Get_rank()`) to select which portion of A this process owns. The block-cyclic pattern ensures that as A rotates through the inner ring, each inner step presents the A block that aligns with the current B tile that the outer ring is pointing at.

**`block_cyclic_distribution` — why it appears in several algorithms:**

Some algorithms require a rank to own a non-contiguous set of columns (or rows) of a matrix. For example, with `px=2, py=3, P=6`, a standard row-major distribution gives each rank a contiguous `(m/2, k/3)` block of A. But certain ring patterns require rank `r` to own columns `{r%px, r%px + px, r%px + 2*px, ...}` — every `px`-th column starting at column `r%px`. `block_cyclic_distribution` concatenates these non-contiguous columns into a single local tile.

**`flatten_gather` (Alg 11, R11):**  
Most algorithms have each rank own exactly one tile of C, which is returned from `get_local_indices` as a single `(tile, (row, col))` pair. In Alg 11 (`outer=RS_C_COL, inner=AG_B_COL`), the C distribution is `block_cyclic`, so each rank owns multiple non-contiguous C tiles. `get_local_indices` returns a list of `(tile, (row, col))` pairs. `flatten_gather=True` tells `Gemm2D.setup_and_run()` to flatten the gathered list of lists before reassembling C.

### 6.4 `make_compute_fn`

Always: `lambda *args: _make_compute_fn(key, *args)`

This just binds the algorithm key so the factory knows which `GEMM_2D_INNER_CONFIGS` entry to use. All the real logic is in `_make_compute_fn`.

---

## 7. `Gemm2D`: Wiring It All Together

```python
class Gemm2D:
    def __init__(self, first, second):
        t1 = (first.config.matrix_communicated, first.config.subtile_scheme)
        t2 = (second.config.matrix_communicated, second.config.subtile_scheme)
        key = (t1, t2)  # ordered: first=outer, second=inner
```

**Why `first` is always outer:** By convention, the outer loop runs over the larger communicator (the one with more cross-rank communication for AG, or equivalently the first factor in the algorithm name). The key is ordered, so `Gemm2D(A_COL, B_COL)` and `Gemm2D(B_COL, A_COL)` are different algorithms. There is no ambiguity.

**`setup_and_run()` — communicator creation (lines 1321–1326):**

```python
if self._outer_is_ag:
    outer_comm = remainder_communicator(comm, group_param, rank)
    inner_comm = nearby_rank_communicator(comm, group_param, rank)
else:
    outer_comm = nearby_rank_communicator(comm, group_param, rank)
    inner_comm = remainder_communicator(comm, group_param, rank)
```

The AG/RS status of the outer loop determines the communicator assignment. This is a fixed convention: AG-outer algorithms always use `remainder` for the outer ring; RS-outer algorithms use `nearby`. The reason is that the `remainder_communicator` groups ranks that are `group_param` apart (e.g., ranks 0, py, 2*py in a remainder-py split), which corresponds to ranks in the same "column" of the process grid. AG algorithms that distribute A column-wise want ranks in the same column to communicate. RS algorithms distribute differently.

**`run()` — how the outer loop is invoked (lines 1364–1372):**

```python
outer_mc = self.outer_gemm1d.config.matrix_communicated
outer_ct = GEMM_2D_OUTER_CURRENT_TILES[outer_mc]

return self.outer_gemm1d.run(
    A, B, C, outer_comm, outer_size, outer_rank,
    compute_fn=compute_fn,
    current_tiles_override=outer_ct,
    set_c_override=_noop_set_c,
)
```

Three overrides are applied to the outer `Gemm1D.run()`:

1. `compute_fn=compute_fn` — replaces `np.matmul(A_curr, B_curr)` with the inner loop closure
2. `current_tiles_override=outer_ct` — replaces the outer loop's normal subtiling with the 2D version that returns full matrices and a zero C
3. `set_c_override=_noop_set_c` — prevents the outer loop from writing to C (inner loop owns C writes)

The outer loop is never aware it is doing 2D work. It runs its standard ring protocol, calls `compute_fn` at each step, and the no-op set_c discards its C writes. The inner loop runs in full inside each `compute_fn` call.

---

## 8. Deep Dives: Four Representative Algorithms

### 8.1 Alg 1: `outer=AG_A_COL, inner=AG_A_ROW` (simplest case)

**What it computes:** Both loops communicate A. The outer ring distributes column-strips of A across `py` processes (each rank "sees" different columns of A via the outer ring). Inside each outer step, the inner ring distributes row-strips of A across `px` processes. Together they cover all `px * py` rank pairs.

**Distribution:** A is `row_major_distribution` — rank `r` owns the `(r//py, r%py)` tile of A (shape `m/px × k/py`). B is `pure_column_distribution` — each rank owns one column-band of B. C is `pure_column_distribution` — same layout as B.

**Inner config:**
- `make_buffer=DoubleBuffer(np.copy(A_outer))`: A fresh copy of the current outer A-strip becomes the inner buffer. `np.copy` because A_outer is a view that will change when the outer buffer swaps.
- `persistent_buffer=False`: New buffer each outer step, since A_outer changes and it's used as the initial buffer value.
- `loopback=False`: After `inner_size - 1` sends, the inner loop has seen all px A-tiles. No need to loop back.
- `tiles.C_curr = get_subtile(C, os_, 1, oi, 0)`: At outer step `oi`, use row-band `oi` of C (since there are `os_ = py` row-bands). This is constant across all inner steps — the entire row-band is the output for this outer step.
- `set_c = set_subtile(C, Ct, os_, 1, oi, 0)`: Write result back into the same row-band.
- `make_C_inner=None`: Inner writes directly to C's row-band.
- `rs_final=None`: Not needed.

**Execution trace (P=6, px=2, py=3, rank=0):**
- Outer ring: remainder-3 communicator → groups {0,3}, {1,4}, {2,5}. Rank 0 is in group {0,3}.
- Inner ring: nearby-3 communicator → groups {0,1,2}, {3,4,5}. Rank 0 is in group {0,1,2}.
- Outer step 0 (oi=0): A_outer = rank 0's starting A-strip. Inner loop runs over {0,1,2}: A-strips rotate, B stays fixed at rank's column-band, result writes to C[row_band_0, :]. `compute_fn` returns zeros.
- Outer step 1 (oi=1): A_outer = A-strip received from rank 3 (rank 3's starting A). Same inner loop, result writes to C[row_band_1, :].
- After both outer steps: C has been fully computed for rank 0's C column-band.

---

### 8.2 Alg 6: `outer=AG_A_ROW, inner=AG_B_COL` (persistent buffer with loopback)

**Why it needs a persistent buffer:** The inner loop rotates ALL of B across the inner ring. B is not a subtile per outer step — the full local B matrix rotates through `inner_size` positions at each outer step. With `loopback=True`, B returns to its origin after each inner loop. The buffer must persist because the DoubleBuffer object tracks which internal array is currently valid, and after an odd number of swaps, `current_buffer` points to the second array (not the first). Recreating `DoubleBuffer(B)` would point `current_buffer` to `first_buffer = B`, which has correct data only if `inner_size` is even.

**Why loopback is needed:** Without loopback, after `inner_size - 1` swaps, each rank's buffer holds the tile from its ring neighbor, not its original tile. The next outer step would start with B in the wrong position. With loopback (`inner_size` swaps), B has traveled a full ring and each rank's buffer again holds its original tile.

**Distribution coupling:** A is `alternating_row_distribution(M, px, py, rank)` — each rank's local A row-strip corresponds to its outer ring position. C is `col_major_distribution(py, px, rank)` — tiles are ordered column-major with py as the "column" dimension, matching the output subtile addressing `get_subtile(C, os_, sz, oi, idx)` where `os_=py`, `sz=px`.

**The `tiles` lambda:**

```python
'tiles': lambda A_outer, B_outer, oi, os_: CurrentTiles(
    A_curr=lambda A, B, C, buf, idx, sz: A_outer,
    B_curr=lambda A, B, C, buf, idx, sz: buf.get_buffer(),
    C_curr=lambda A, B, C, buf, idx, sz: get_subtile(C, os_, sz, oi, idx),
),
```

C is addressed by BOTH `oi` AND `idx` — the outer step selects the row-group and the inner step selects the column within that group. This gives a unique C subtile for every `(oi, idx)` pair, covering all `px * py` subtiles of C exactly once across all outer+inner steps.

---

### 8.3 Alg 8: `outer=AG_A_ROW, inner=RS_C_COL` (AG outer + RS inner, requires `rs_final`)

**The challenge:** The outer loop communicates A row-strips. At each outer step, this rank works on a different K-block of A. The inner loop is ReduceScatter over C columns — it distributes partial sums of a single C column-strip across `inner_size` ranks. After the inner RS loop finishes, the result is one fully-reduced C subtile.

**`make_buffer=None`:** RS inner loops don't need an outer-controlled buffer. `Gemm1D.run()` creates its own `AccumulationBuffer(C_inner)` via the 1D config.

**`make_C_inner`:**

```python
'make_C_inner': lambda C, oi, is_, os_:
    np.zeros(get_subtile_shape(C, os_, 1), dtype=MATRIX_DTYPE),
```

Creates a zero array shaped `(m/os_, n)` — a zero column-strip of C. This is the accumulator for the inner RS loop. Size `(m/os_, n)` because after `os_` outer steps, C is split into `os_` row-bands, and this outer step is working on row-band `oi`. But note the inner RS needs to accumulate `inner_size` partial sums — the zero array is the starting point for that accumulation.

**`rs_final`:**

```python
'rs_final': lambda result, C, oi, is_, os_:
    set_subtile(C, result + get_subtile(C, os_, 1, oi, 0), os_, 1, oi, 0),
```

After the inner RS loop completes, `result` is the accumulated partial sum for this rank's portion of C (the RS output). This partial sum must be added to the already-accumulated value in the real C (from previous outer steps, since we're doing AG outer). `rs_final` reads the current value from C's row-band `oi`, adds the new partial sum, and writes it back.

**Return value from `compute_fn`:** After calling `rs_final`, returns `np.zeros(C.shape)`. The outer AG loop receives zeros from every outer step, computes `C_tmp = zeros + zeros = zeros`, and discards them via `_noop_set_c`. The outer loop acts purely as a mechanism to rotate A and call `compute_fn` repeatedly.

---

### 8.4 Alg 4: `outer=RS_C_COL, inner=AG_A_COL` (RS outer + AG inner, persistent + loopback)

**The challenge:** The outer loop communicates C column-strips (ReduceScatter). Each outer step accumulates partial sums for C. The inner loop communicates all of A (AllGather). A must rotate through ALL inner ranks continuously across outer steps — it's a persistent ring.

**Why persistent AND loopback together:** The outer RS ring rotates C. But A is being rotated in the inner ring at the same time. After each inner loop (with loopback), A returns to its origin rank in the inner ring. But the outer RS step has moved to the NEXT outer rank. The persistent buffer ensures A's "current" pointer is maintained across both loops. The loopback ensures A physically returns to the correct inner rank after each inner loop, so the persistent buffer's data is consistent at the start of each outer step.

**`make_C_inner`:**

```python
'make_C_inner': lambda C, oi, is_, os_:
    np.zeros(C.shape, dtype=MATRIX_DTYPE),
```

Creates a zero array of the full C shape. This is necessary because the outer RS loop accumulates the `compute_fn` return values. Each outer step, the inner AG loop computes a partial sum for the full local C (not just a subtile). This partial sum is returned to the outer RS ring which routes it to the correct rank.

**The outer RS control flow:** Looking at `Gemm1D.run()` for RS outer:

```
i=0: C_curr = zeros (from OUTER_CURRENT_TILES), compute_fn → partial_sum_0
     C_tmp = partial_sum_0 + zeros = partial_sum_0
     send C_tmp (partial_sum_0) to next outer ring member
i=1: C_curr = received partial_sum from previous rank's step 0
     compute_fn → partial_sum_1
     C_tmp = partial_sum_1 + C_curr (accumulating two partial sums)
     send C_tmp
...
i=last: C = C + C_tmp (final accumulated result written into local C)
```

Each outer step, the inner AG computes a full partial sum of `A_row_segment × B`. These are routed around the outer RS ring, summed, and the final fully-reduced partial sum lands at the rank that owns that C tile. The outer RS loop does the routing and accumulation; it doesn't care that `compute_fn` is running a whole inner loop.

**`tiles.B_curr = get_subtile(B, sz, os_, idx, oi)`:** This is the most complex tile address in the codebase. Both `idx` (inner step) and `oi` (outer step, via closure) index into B. B is split into a `sz × os_` grid (where `sz = inner_size` and `os_ = outer_size`). At outer step `oi`, inner step `idx`, the matmul uses B's tile at row `idx`, column `oi`. This ensures that as both loops advance, each unique `(K-inner, K-outer)` block of B is paired with exactly the right K-block of A that the inner ring delivers at that step.

---

## Summary: How to Read Any Algorithm Entry

1. **Find the key:** `((outer_matrix, outer_subtile), (inner_matrix, inner_subtile))` in both `GEMM_2D_INNER_CONFIGS` and `GEMM_2D_ALGORITHMS`.

2. **Determine communicators:** `group_param_is_py` gives the group size. AG outer → outer=remainder, inner=nearby. RS outer → reversed.

3. **Check persistence:**  
   - `persistent_buffer=True` + `loopback=True`: the inner buffer rotates continuously across all outer steps and must physically loop back to its origin after each inner run.
   - `persistent_buffer=False` + `loopback=True`: buffer is subtile-based and needs loopback to restore the parent matrix, but a new buffer is created each outer step.
   - `persistent_buffer=False` + `loopback=False`: fully independent inner loops, no continuity between outer steps.

4. **Trace C management:**
   - `make_C_inner=None`: inner writes directly to real C via `set_c`. Outer is AG.
   - `make_C_inner` + `rs_final`: inner is RS, outer is AG. `rs_final` merges the RS result into C after each inner loop.
   - `make_C_inner` + `rs_final=None`: outer is RS. `compute_fn` returns the partial sum to the outer RS ring.

5. **Decode `tiles`:** The `C_curr` lambda tells you exactly which subtile of C is being computed at each `(oi, idx)` pair. Match it against `set_c` to verify read and write addresses are identical.
