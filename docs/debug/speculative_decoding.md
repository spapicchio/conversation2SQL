
## Table of Contents

1. [[#1 — The Problem: Why Generation is Slow]]
2. [[#2 — Speculative Decoding: Core Mechanics]]
3. [[#4 — Multi-Token Prediction (MTP)]]
4. [[#5 — EAGLE / EAGLE-3]]
5. [[#6 — DFlash: State of the Art]]

---
## 1 - The Problem: Why Generation is Slow

### 1.1 The Memory-Bandwidth Wall

Modern GPUs are built for *training* workloads — dense matrix-matrix multiplications with high arithmetic intensity. Inference is a fundamentally different regime.

**Arithmetic intensity** is defined as the ratio of compute operations to memory bytes transferred:
$$I = \frac{\text{FLOPs}}{\text{bytes transferred}}$$
A GPU's **roofline** tells you which resource limits performance. For a given operation:
- If $I >$ (peak FLOPs / peak bandwidth) → **compute-bound** (GPU is the bottleneck)
- If $I <$ (peak FLOPs / peak bandwidth) → **memory-bandwidth-bound** (HBM is the bottleneck)

For reference an NVIDIA H100-NVL:

| Metric                 | Value               |
| ---------------------- | ------------------- |
| Peak FP16 FLOPs        | ~1700 teraFLOPS     |
| Peak HBM bandwidth     | ~3.9 TB/s           |
| **Roofline threshold** | **~436 FLOPs/byte** |

At **batch size $B$**, FP16 precision and *d*=4096, the input is a matrix $H \in \mathbb{R}^{B \times 4096}$ and the operation is a matrix-matrix multiply:
$$\text{FLOPs} = 2 \times B \times 4096^2, \quad \text{Bytes} \approx 4096^2 \times 2 \quad \text{(weights loaded once)}$$
$$I = \frac{2 \times B \times 4096^2}{4096^2 \times 2} = B \text{ FLOPs/byte}$$
To reach the compute roofline ($I = 436$) you need $B \approx 436$, so you are memory-bound.

> **Source:** Pope et al., *Efficiently Scaling Transformer Inference*, MLSys 2023. 

---
### 1.2 The Sequential Constraint

Even with infinite compute, you cannot parallelize autoregressive generation. The reason is structural:
$$P(y_1, y_2, \ldots, y_m \mid x) = \prod_{i=1}^{m} P(y_i \mid x, y_1, \ldots, y_{i-1};\, \theta)$$
Token $y_i$ is **causally dependent** on all previous outputs. You cannot compute position $i$ before position $i-1$ is complete. With $m$ output tokens, you are forced to take exactly $m$ sequential forward passes.
**The KV cache** solves the *compute redundancy* problem (not recomputing past attention) but does nothing about sequentiality:
- At step $i$, load $K_j, V_j$ for all $j < i$ from cache → compute attention → store $K_i, V_i$
- Cost per step: $O(L \cdot d^2)$, constant, but still **one step per token** (L=Num Layers)

> **Source:** Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.

---
### 1.3 Two Independent Problems, One Solution Family

```
┌─────────────────────────────────────────────────────────────┐
│ PROBLEM 1: Each step is slow (memory-bound)                 │
│ PROBLEM 2: Steps cannot be parallelized (sequential)        │
│                                                             │
│ SOLUTION: Generate candidate tokens cheaply (draft),        │
│ verify many at once in parallel (target)                    │
└─────────────────────────────────────────────────────────────┘
```
  
This is the core idea behind **Speculative Decoding** — exploit the asymmetry between generation (sequential, slow) and verification (parallel, fast).

---
## 2 - Speculative Decoding: Core Mechanics

Generate candidate tokens cheaply (draft), verify many at once in parallel (target).
This is the core idea is to exploit the asymmetry between generation (sequential, slow) and verification (parallel, fast).

### 2.1 The Fundamental Asymmetry

> **Generating** $\gamma$ tokens takes $\gamma$ sequential target model forward passes.
> **Verifying** a proposed sequence of $\gamma$ tokens takes **1** parallel target forward pass.

Why? Because the target model only needs to compute the conditional distributions $P(y_i \mid y_{<i})$ for all $i$ simultaneously,  this is a standard forward pass over a sequence of $\gamma+1$ tokens, which the transformer executes in parallel via its attention mechanism.
  
> **Source:** Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.

---
### 2.2 The Draft-Verify Loop (Full Algorithm)

**Inputs:** $M_p$ (target model), $M_q$ (draft model), $prefix$, $\gamma$ (speculation width)

**Step 1 — Draft:** sample $\gamma$ guesses from $M_q$ autoregressively
$$\text{for } i = 1 \text{ to } \gamma: \quad q_i(x) \leftarrow M_q(prefix + [x_1, \ldots, x_{i-1}]), \quad x_i \sim q_i(x)$$

**Step 2 — Verify:** run $M_p$ in parallel over all draft tokens in one forward pass
$$p_1(x), \ldots, p_{\gamma+1}(x) \leftarrow M_p(prefix),\, \ldots,\, M_p(prefix + [x_1, \ldots, x_\gamma])$$

**Step 3 — Find first rejection:** sample $r_1, \ldots, r_\gamma \sim U(0,1)$ and compute
$$n \leftarrow \min\!\left(\{i - 1 \mid 1 \leq i \leq \gamma,\; r_i > \tfrac{p_i(x_i)}{q_i(x_i)}\} \cup \{\gamma\}\right)$$
- If $p_i(x_i)≥q_i(x_i)p_i(x_i) \geq q_i(x_i) p_i​(x_i​)≥q_i​(x_i​)$→ the ratio is $\geq1 → r_i\leq 1$ always → **always accept**. The target model agrees with or is more confident than the draft on this token.
- If $p_i(x_i)≪q_i(x_i)$ → the ratio is close to 0 → almost certainly reject. The draft model is overconfident on a token the target model considers unlikely.

$n$ is the position where the first rejection occurs.

**Step 4 — Adjust distribution at rejection point:**
$$p'(x) \leftarrow p_{n+1}(x)$$
$$\text{if } n < \gamma: \quad p'(x) \leftarrow \text{norm}\!\left(\max\!\left(0,\; p_{n+1}(x) - q_{n+1}(x)\right)\right)$$

**Step 5 — Sample bonus token and return**
$$t \sim p'(x)$$
$$\text{return } prefix + [x_1, \ldots, x_n, t]$$
> Always returns $n + 1$ tokens: $n$ accepted draft tokens plus 1 token from $M_p$.  
> Therefore $\tau \in [1, \gamma + 1]$ — minimum 1 (all drafts rejected), maximum $\gamma + 1$ (all accepted).

---
### 2.3 Speedup Analysis 

**Expected tokens per cycle.** Under the i.i.d. acceptance rate assumption with $\alpha = E(\beta)$ (where $\beta$ is the acceptance rate mentioned above), the number of tokens produced by one run of previous algorithm:
$$E(\#\;\text{generated tokens}) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$
This is bounded above by $\frac{1}{1-\alpha}$ as $\gamma \to \infty$. In practice, $\alpha \in [0.5, 0.9]$ for well-matched mode pairs.

**Walltime improvement** . Let $c$ be the ratio between the time for a single run of $M_q$ and a single run of $M_p$. The expected improvement factor in total walltime is:
$$\boxed{\frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}}$$
The two levers are explicit here:
- **Maximize $\alpha$**: better draft quality → more tokens accepted per cycle
- **Minimize $c$**: cheaper draft model → less overhead per cycle

**When does it help?**  A necessary and sufficient condition for any speedup to exist is:
$$\alpha > c$$
The draft model must have a higher acceptance rate than its cost ratio relative to the target. If this holds, the minimum guaranteed improvement is:
$$\frac{1 + \alpha}{1 + c}$$
**How to choose $\gamma$** . Given $\alpha$ and $c$, the optimal $\gamma$ maximizes the walltime improvement formula above. Since $\gamma$ is an integer it can be found numerically. Two practical observations from the paper:
- For $c \approx 0$ (negligible draft cost), the improvement approaches $\frac{1}{1-\alpha}$,  bounded only by acceptance rate
- Varying $\gamma$ dynamically during inference based on predicted $\beta$ can yield up to ~60% additional improvement over a fixed $\gamma$

> **Source:** Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.

---
## 3 - Multi-Token Prediction (MTP)

### 3.1 What MTP Is (and Isn't)

MTP is **a training objective**, not an inference algorithm. It does not require a separate draft model, a verification loop, or any inference-time modification.

> Standard training: at each position $t$, predict the next token $y_{t+1}$
> MTP training: at each position $t$, predict the next $n$ tokens $y_{t+1}, y_{t+2}, \ldots, y_{t+n}$

The model learns to encode information about multiple future tokens in its hidden states. This has two effects:

1. **Better representations:** richer training signal → better model quality (12% gain on HumanEval, 17% on MBPP for 13B models)
2. **Inference speedup:** the extra heads can be repurposed as a built-in draft model at inference time

> **Source:** Gloeckle et al., *Better & Faster Large Language Models via Multi-Token Prediction*, ICML 2024. 

---
### 3.2 Architecture: Shared Trunk + Independent Heads

```

Input: x₁, x₂, …, xₜ

     │

┌────▼─────────────────────────────────┐

│ Shared Transformer Trunk             │ (main model, all L layers)

└────┬─────────────────────────────────┘

     │ hidden state hₜ ∈ ℝᵈ

     │

┌────▼────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐

│ Head 1  │ │ Head 2  │ │ Head 3  │ │ Head 4  │ (independent)

└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘

     │           │           │           │

  P(yₜ₊₁)      P(yₜ₊₂)     P(yₜ₊₃)      P(yₜ₊₄)

```


**Training loss** (auxiliary, summed over all heads):
$$\mathcal{L}_\text{MTP} = -\sum_{k=1}^{n} \lambda_k \sum_{t} \log P_k(y_{t+k} \mid x_{\leq t})$$

where $P_k$ is the $k$-th prediction head and $\lambda_k$ are optional decay weights.

**Memory efficiency trick (Gloeckle et al.):** Rather than materializing all $n$ logit vectors simultaneously (cost $O(nV)$ where $V$ = vocabulary size), each head performs its forward + backward pass sequentially, accumulating gradients at the trunk. Peak memory stays at $O(V + d)$ instead of $O(nV + d)$.

> **Source:** Gloeckle et al. 2024.

---
### 3.4 MTP × SD: The Connection

When MTP heads are used as draft model at inference time, this is exactly speculative decoding — the main model is the target, the heads are the draft:

```
┌──────────────────────────────┐

│ Main model (target)          │ ← verifies

└──────────┬───────────────────┘

           │ hidden states

┌──────────▼───────────────────┐

│ MTP heads (draft)            │ ← proposes γ tokens

└──────────────────────────────┘
```

**Key insight:** because the MTP heads were trained *jointly* with the main model on the same data, their output distribution is extremely well-aligned with the target. In terms of the speedup analysis from Section 2.3:

- High alignment → high $\alpha$ (acceptance rate per token)
- High $\alpha$ → $E(\#\text{ generated tokens}) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$ grows
- Low $c$ (heads are tiny compared to $M_p$) → denominator $(\gamma c + 1)$ stays close to 1
- Combined: walltime improvement $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$ is large

In DeepSeek-V3, the MTP head acceptance rate exceeds 80%, yielding **1.8× throughput speedup** in practice.

No separate model to load, no separate training run, no checkpoint management.
The draft is already inside the model.

---
## 4 - EAGLE / EAGLE-3

### 4.1 The Core Idea: Feature-Level Conditioning

In standard speculative decoding, $M_q$ is a separate smaller model that generates tokens using only its own weaker parameters, independently of what $M_p$ has computed.It has no access to $M_p$'s internal state. 

EAGLE takes a different approach. When $M_p$ processes a sequence $x_1, \ldots, x_t$, it produces at the last layer a hidden state $h^{(L)}_t \in \mathbb{R}^d$. This is the exact vector $M_p$ uses to compute its own next-token prediction: $$P(x_{t+1} \mid x_{\leq t}) = \text{softmax}(\text{LMHead}(h^{(L)}_t))$$ $h^{(L)}_t$ is the result of processing the full sequence through all $L$ layers of $M_p$ : every attention pattern, every MLP transformation. It implicitly encodes information about multiple future tokens, because to predict $x_{t+1}$ accurately the model must have internally represented what is likely to follow. 

EAGLE gives $M_q$ direct access to $h^{(L)}_t$. Rather than reconstructing the context from scratch with inferior capacity, the draft model builds directly on top of $M_p$'s own reasoning. This is why EAGLE achieves substantially higher $\alpha$ than a classic small draft model of comparable size.

> **Source:** Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024.

---
### 4.2 EAGLE-1 Architecture

**Draft model input at step $i$:**
$$\text{input}_i = \bigl[h^{(L)}_{i-1};\; \text{embed}(\tilde{y}_{i-1})\bigr] \in \mathbb{R}^{2d}$$

A linear projection maps this to $\mathbb{R}^d$, then one transformer layer produces the next draft token.

**Draft model:** 1 transformer layer + shared embedding and LM head (frozen from target).

```
Target model ──→ h^(L)_{i-1} ─┐

                              ├──→ [concat] ──→ FC ──→ 1-layer Transformer ──→ ỹᵢ

embed(ỹᵢ₋₁) ─────────────────-┘
```

**Why 1 layer?** Adding depth to the draft model affects two things simultaneously:

**Cost** grows linearly. Each of the $\gamma$ draft steps is sequential, step $i$ depends on the output of step $i-1$, so they cannot be parallelized. Total drafting cost is:
$$T_\text{draft} = \gamma \cdot t_\text{step}$$
Adding one layer raises $t_\text{step}$, and that increase is multiplied by $\gamma$. The more tokens you speculate, the more expensive depth becomes.

**Quality gain saturates.** $h^{(L)}$ is injected once, at the input. After layer 1 it has been transformed by attention and MLP, the target signal is diluted. After layer 2 it is further diluted. The improvement in $\alpha$ from each additional layer rapidly decreases because the guidance from $M_p$ is no longer intact at deeper layers.

The result: cost grows linearly with depth while quality gain quickly plateaus. **1 layer is the only depth where the gain in $\alpha$ justifies the increase in $T_\text{draft}$.**

---
### 4.3 EAGLE-2: Adaptive Draft Trees

**The problem with flat drafting.** EAGLE-1 generates a single sequence $\tilde{y}_1, \tilde{y}_2, \ldots, \tilde{y}_\gamma$. 
This is a single bet on one continuation. If $\tilde{y}_1$ is wrong, the entire chain is discarded: a rejection at position 1 costs you all $\gamma - 1$ subsequent draft tokens. The expected tokens generated per cycle depends heavily on getting early positions right.

**EAGLE-2's solution: draft a tree instead of a sequence.**

At each step, instead of sampling one token, the draft model proposes the top-$k$ candidates. This creates a tree where each node is a possible draft token and each path from root to leaf is one candidate sequence:

```

Step 1: draft model proposes top-3 candidates from prefix

prefix ──→ draft model ──→ ỹ₁ₐ (p=0.6); ỹ₁ᵦ (p=0.3); ỹ₁꜀ (p=0.1)

Step 2: each candidate is extended by top-2

      ỹ₁ₐ ──→ { ỹ₂ₐₐ (p=0.5)
              { ỹ₂ₐᵦ (p=0.3)

      ỹ₁ᵦ ──→ { ỹ₂ᵦₐ (p=0.4)
              { ỹ₂ᵦᵦ (p=0.2)

      ỹ₁꜀ ──→  (pruned — low confidence)
```

Full tree (7 nodes):

```
                       ┌──→ ỹ₂ₐₐ
          ┌──→ ỹ₁ₐ ────┤
          │            └──→ ỹ₂ₐᵦ
          │
prefix ───┼──→ ỹ₁ᵦ ────┬──→ ỹ₂ᵦₐ 
          │            └──→ ỹ₁꜀ (pruned)
          └──→ ỹ₂ᵦᵦ    
```

**Why this raises $\alpha$.** A flat sequence bets everything on the single most likely continuation. The tree covers the top-$k$ most likely continuations at each step: the probability that at least one path matches what $M_p$ would have generated is substantially higher. More of $M_p$'s probability mass is covered.

**Verification in one forward pass.** $M_p$ verifies all paths simultaneously using **sparse tree attention**: each node attends only to its own ancestors (its path from root), never to sibling branches.

```
ỹ₂ₐₐ attends to: prefix, ỹ₁ₐ (its ancestors only) 
ỹ₂ₐᵦ attends to: prefix, ỹ₁ₐ 
ỹ₂ᵦₐ attends to: prefix, ỹ₁ᵦ 
ỹ₂ᵦᵦ attends to: prefix, ỹ₁ᵦ
```

$M_p$ runs once over all nodes. The best accepted path is kept. This is still a single target forward pass (cost is $O(1)$ regardless of tree size), though larger trees do increase the number of tokens in that pass.

**Adaptive tree construction.** EAGLE-2 does not use a fixed branching factor. It allocates the total node budget (up to 60 nodes) dynamically:
- High confidence position (top-1 probability is high) → fewer branches needed, concentrate budget downstream
- Low confidence position → more branches to cover the distribution

This avoids wasting nodes on positions where the draft is already certain, and concentrates them where uncertainty is high and coverage matters most.

> **Source:** Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, 2024.

---
### 4.4 EAGLE-3: Training-Time Test


**The train/test mismatch.** In EAGLE-1 and EAGLE-2, the draft model is trained with teacher forcing: at each training step, the input to the draft model is always the real $h^{(L)}$ from $M_p$ computed on the ground-truth token. The model is never exposed to its own mistakes during training.

At inference however, the draft model runs autoregressively, each step receives $h^{(L)}$ computed from the *previous draft token*, which may itself be wrong. Errors compound:

```
Training (teacher forcing): 
step 1: input = h^(L) from ground-truth x₁ → predict x₂ ✓ clean 
step 2: input = h^(L) from ground-truth x₂ → predict x₃ ✓ clean 
step 3: input = h^(L) from ground-truth x₃ → predict x₄ ✓ clean

Inference (autoregressive): 
step 1: input = h^(L) from real prefix → produce ỹ₁ 
step 2: input = h^(L) computed from ỹ₁ → produce ỹ₂ ⚠ depends on ỹ₁ 
step 3: input = h^(L) computed from ỹ₁, ỹ₂ → produce ỹ₃ ⚠ errors accumulate
```

The model was trained on clean inputs at every step but at inference it sees increasingly degraded inputs. The further into the chain, the larger the distribution gap between training and inference, which is precisely where accurate predictions matter most for a long acceptance run.

**EAGLE-3's fix: simulate the inference chain during training.**
At each training step, EAGLE-3 unrolls the full $\gamma$-step drafting process exactly as it would happen at inference:

```
Training-time test (γ = 3):

step 1: feed real h^(L) → produce ỹ₁ 
↓ 
step 2: feed h^(L) computed from ỹ₁ → produce ỹ₂ 
↓ 
step 3: feed h^(L) computed from ỹ₁,ỹ₂ → produce ỹ₃

with loss computed over all 3 steps jointly
```

The model is now trained on the same distribution it will encounter at inference: including the degraded inputs at later steps. This forces it to learn robust multi-step behaviour rather than single-step extrapolation on clean inputs.

The practical effect is a direct improvement in $\alpha$ at later positions in the draft chain, exactly where EAGLE-1 and EAGLE-2 degrade most.

> **Source:** Li et al., *EAGLE-3: Scaling Up Inference Acceleration of Large Language Models via Training-Time Test*, 2025.

---
### 4.5 The Feature Dilution Problem (Why EAGLE Can't Scale Depth)

EAGLE injects the target hidden state $h^{(L)}$ only **at the input** of the draft model. If we add more layers to the draft model, the target signal must survive through successive attention and MLP operations:

```
Layer 1: strong target signal (directly from input)
Layer 2: diluted (after 1 × attn + MLP)
Layer 3: further diluted
Layer 5: essentially lost
```

Adding depth yields diminishing returns in $\alpha$,  it quickly plateaus because the target signal guiding deeper layers is no longer intact, while $t_\text{step}$ grows linearly, increasing $T_\text{draft} = \gamma\cdot t_\text{step}$. From the speedup formula:
$$\frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$
$\alpha$ stops improving with depth while $c$ (the cost ratio of $M_q$ relative to $M_p$) keeps growing. The denominator increases while the numerator plateaus. **This caps EAGLE at 1 layer.**

The natural fix is re-injecting the target signal at every layer so it never dilutes. This is precisely what DFlash solves in Section 5.

---
## 5 - DFlash
### 5.1 Block Diffusion Drafting

DFlash replaces autoregressive drafting with **block diffusion**: all $\gamma$ positions are decoded in parallel in a single forward pass.

```
EAGLE-3 drafting (γ=4): DFlash drafting (γ=4):

t=1: → ỹ₁ Single pass:
t=2: ỹ₁ → ỹ₂ [MASK][MASK][MASK][MASK]
t=3: ỹ₁ỹ₂ → ỹ₃ ↓ (one forward pass)
t=4: ỹ₁ỹ₂ỹ₃ → ỹ₄ [ỹ₁][ỹ₂][ỹ₃][ỹ₄]

Cost: 4 × t_step Cost: t_parallel ≈ t_step
```

Within the block, the draft model uses **bidirectional attention** (all masked positions see each other). This is why a single denoising step is enough — bidirectional context makes the prediction problem much easier than left-to-right autoregressive generation.

$$T_\text{draft}^\text{DFlash} = t_\text{parallel} \approx \text{const}(\gamma) \quad \text{for } \gamma \lesssim 32$$

GPU hardware executes this parallel pass at nearly identical cost to a single-token pass because both are memory-bandwidth-bound — the bottleneck is loading model weights once, not the number of positions processed.

---
### 5.2 KV Injection: The Key Innovation

DFlash extracts hidden states from **5 uniformly-sampled layers** of the target model (from layer 2 to layer $L-2$), concatenates them, and projects to a compact *target context feature* $c$:
$$c = W_\text{proj} \cdot \bigl[h^{(\ell_1)};\, h^{(\ell_2)};\, \ldots;\, h^{(\ell_5)}\bigr] \in \mathbb{R}^{d_\text{draft}}$$
This feature is then **injected into every draft layer's Key and Value projections**:
$$K^\ell_\text{aug} = \bigl[W_K^\ell \cdot c\;;\; W_K^\ell \cdot e_\text{masked}\bigr]$$
$$V^\ell_\text{aug} = \bigl[W_V^\ell \cdot c\;;\; W_V^\ell \cdot e_\text{masked}\bigr]$$

At every draft layer, attention queries look at both the target context and the masked tokens. The signal stays **constant strength at every depth**.

Compare to EAGLE:

|                            | EAGLE-3          | DFlash               |
| -------------------------- | ---------------- | -------------------- |
| Where target signal enters | Input layer only | **Every layer (KV)** |
| Signal at layer $k$        | Decays with $k$  | Constant             |
| Consequence                | 1 layer optimal  | 5 layers optimal     |
| τ (Math500, Qwen3-8B)      | 3.49             | **7.87**             |

---
### 5.3 Training Innovations

**Position-dependent loss decay.** In speculative decoding, an error at position $k=1$ invalidates all subsequent tokens. DFlash weights the training loss to reflect this:

$$w_k = \exp\!\left(-\frac{k-1}{\gamma}\right)$$

For $\gamma=16$: $w_1 = 1.0$, $w_4 \approx 0.83$, $w_8 \approx 0.61$, $w_{16} \approx 0.37$.

Position 1 receives **2.7× more gradient signal** than position 16.

**Random anchor sampling.** Instead of fixed block boundaries, DFlash randomly samples anchor tokens from the response as starting positions for each masked block. This exposes the draft model to diverse relative positions, acting as data augmentation. Result: +13.6% speedup on Math500 vs standard block construction.

**Efficient long-context training via Flex Attention.** Multiple blocks are concatenated into a single sequence. The attention mask enforces: bidirectional within each block, no attention across blocks, full attention to target context features. Implemented via PyTorch Flex Attention for efficient sparse pattern execution.

---
### 5.4 Measured Results

On Qwen3-8B, Transformers backend, greedy decoding:

|                      | EAGLE-3 (tree-60) | DFlash (block-16) | DFlash advantage |
| -------------------- | ----------------- | ----------------- | ---------------- |
| Avg speedup (temp=0) | 2.02×             | **4.86×**         | **+2.4×**        |
| Avg speedup (temp=1) | 1.88×             | **4.03×**         | **+2.1×**        |
| Avg τ (temp=0)       | 3.40              | **6.49**          | **+91%**         |
| Peak speedup         | 2.23×             | **6.08×**         | —                |

SGLang production results (Qwen3-8B, B200 GPU):

| Concurrency | DFlash speedup | EAGLE-3 (tree-60) expected  |
| ----------- | -------------- | --------------------------- |
| 1           | 5.1×           | ~2.0×                       |
| 8           | 4.5×           | ~1.3×                       |
| 16          | 3.9×           | **~0.9×** (below baseline!) |
| 32          | 2.8×           | **~0.6×**                   |
