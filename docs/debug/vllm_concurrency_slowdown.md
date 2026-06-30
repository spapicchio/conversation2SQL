# Why higher vLLM concurrency made the eval *slower*

**Date:** 2026-06-05
**Runs compared:** `results/2026-06-05/07-48-20` (concurrency 128) vs `results/2026-06-05/07-59-38` (concurrency 10)
Same model, same data, separate GPUs (no contention). The **only** difference was `concurrency`.

## TL;DR

Counter-intuitively, the run with **10** parallel requests finished much faster than the
one with **128**. Cranking concurrency too high overloaded the GPU's short-term memory
(the "KV cache"), so the server spent almost all its time **re-reading old conversation
text instead of writing new answers**. The fix: keep concurrency moderate (~32) and the
per-request length cap (`max-model-len`) at a realistic 32k instead of 128k.

---

## The evidence

Both servers print live stats. The contrast is stark:

| What the GPU was doing            | 128 concurrency        | 10 concurrency      |
|-----------------------------------|------------------------|---------------------|
| Requests running at once          | 128                    | 10                  |
| **Answer-writing speed / request**| **~2–7 tokens/sec** 🐢 | **~50 tokens/sec** 🚀|
| Time spent re-reading old text    | 7,000–17,000 tokens/s  | 500–2,000 tokens/s  |
| Cache reuse ("prefix hit rate")   | 66% → **falling to 43%** | rising to **84%** |
| Short-term memory (KV cache) used | ~80%                   | ~6%                 |

Each individual answer came out **~10× slower** at concurrency 128. Because the agent
must wait for each answer before taking its next step, the whole run dragged.

---

## Why it happens (plain English)

The model holds every active conversation in a fast scratchpad on the GPU (the **KV
cache**). Normally it caches the start of each conversation so it doesn't re-read it every
turn.

- With **10** conversations, they all fit in the scratchpad. Even when a conversation
  pauses (waiting on a tool or the simulated user), its cached text survives. When it
  resumes, the model picks up instantly. → 84% of the text is reused.

- With **128** conversations, the scratchpad is overcrowded. While a conversation is
  paused, its cached text gets thrown out to make room for others. When it resumes, the
  model must **re-read the entire conversation from scratch**. → cache reuse collapses to
  43%, and the GPU burns its time re-reading (7k–17k tokens/sec) instead of writing new
  answers (which slows to a crawl).

It's like a librarian (the GPU) helping readers (conversations). With 10 readers, everyone
keeps their books on the desk. With 128, the desk overflows, books get reshelved
constantly, and the librarian spends all day fetching the same books over and over.

> Note: this is **eviction**, not a crash. No requests failed; the server just got slow.

---

## The token-size question (does context "overflow"?)

A natural worry: each turn re-sends the whole conversation, and we measured up to
**103k–240k total prompt tokens** per conversation. Doesn't that need a huge context?

**No.** That number is the **sum of ~23 separate API calls**, not one giant request.
Each call only contains the conversation *so far*:

```
Call  1: ~3,000 tokens   (instructions + schema + first question)
Call  2: ~3,400 tokens   (+ one tool result)
   ...                    (grows a little each turn)
Call 23: ~9,000 tokens   (the full conversation)
-------------------------------------------------
SUM    ≈ 103,000          ← the big number, just bookkeeping
BIGGEST single call ≈ 9,000–20,000  ← what actually matters
```

Analogy: saving a 10-page document 23 times "writes" 230 pages of saves, but the document
is still 10 pages. The per-request cap (`max-model-len`) limits the **document size**, not
the total ever written.

**Measured single-request sizes (from 110 conversations):**

| metric                      | average | p95   | max    |
|-----------------------------|---------|-------|--------|
| tokens per single API call  | 4,283   | 6,648 | 7,740  |
| API calls per conversation  | 23      | 33    | 44     |

The largest single request the model ever sees is roughly **10k–20k tokens** — nowhere
near the configured **128k** limit.

---

## Recommendations

1. **`--max-model-len 32768`** (down from 128k). Covers the worst single request (~20k)
   with margin. The current 128k is 6–10× larger than anything we actually use.

2. **Concurrency ≈ 32** (not 10, not 128). The KV cache holds ~552k tokens; at ~6k live
   per request that's ~190k (≈35% used) — fast like the 10 run, but ~3× more throughput.

   | concurrency | live memory needed | of 552k cache | result            |
   |-------------|--------------------|---------------|-------------------|
   | 10          | ~60k               | ~11%          | fast ✓            |
   | **32**      | ~190k              | ~35%          | fast, recommended |
   | 64          | ~380k              | ~70%          | starts to thrash  |
   | 128         | >550k              | >100%         | slow (what we saw)|

3. If you ever *do* need higher concurrency, **add KV cache** (raise
   `--gpu-memory-utilization`, lower `--max-model-len`, or add GPUs / tensor parallelism)
   rather than just raising the request limit.

---

## How to spot this again

Watch the vLLM log. The warning signs of over-concurrency are:

- "Avg generation throughput" (answer-writing) **drops** while "Avg prompt throughput"
  (re-reading) **spikes**.
- "Prefix cache hit rate" **falls** over time instead of rising.
- "GPU KV cache usage" pinned high (70%+).

When healthy (like the conc-10 run): generation steady, prompt throughput low, prefix hit
rate climbing past 80%, KV usage low.
