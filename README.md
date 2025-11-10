# 🧠 BiostatisV6: A Biologically-Inspired Optimizer with Hierarchical Homeostasis

> *An attempt to make a stability-first optimizer inspired by neural homeostasis and multi-scale memory, extending AdamW for heterogeneous architectures.*

---

## 🌱 Intuition

Biological neurons maintain *homeostasis* — stable internal activity levels even when stimuli vary.
**BiostatisV6** brings this principle to deep optimization by monitoring **gradient “activity”** and **coherence** across the model and dynamically adjusting learning behavior to remain within a healthy range.

---
## 🧩 Core Intuition Breakdown

### 1. Global Homeostasis (weak influence – 30%)
   * The optimizer accumulates all gradients into a flat tensor.
   * It computes:
      * **Global energy**: average gradient magnitude 
        $$E_t = \text{mean}(g_t^2)$$
      * **Global coherence**: 
        $$C_t = \text{mean}(|\tanh(g_t)|)$$ 
        which measures gradient activation saturation:
         * Small gradients $(|g| < 1)$ → $\tanh(g) \approx g$: linear, low activation
         * Large gradients $(|g| > 2)$ → $\tanh(g) \to \pm 1$: saturated, high activation
         * $C_t \in [0, 1]$ thus reflects how "excited" or "quiet" the overall network is.
   * A **homeostatic modulation** term gently keeps coherence near the target (≈0.8), preventing either overexcitation or gradient collapse.
   * An **energy feedback** term ensures global gradient magnitude doesn't overshoot or undershoot.


### 2. Local Adaptation (strong influence – 70%)
   * Each parameter performs standard Adam-style momentum updates with bias correction.
   * Two multi-scale exponential memories (decay = 0.9, 0.99) smooth variance across short and long time horizons — analogous to short-term and long-term synaptic memory.
   * **Coherence modulation**: Uses cosine similarity between the momentum and current gradient to align directions (directional coherence).
   * **Selective ascent**: If a parameter's signal-to-noise ratio is low, it adds a small "ascent" term to escape sharp minima.
   * **Local homeostasis**: Per-parameter energy and activation (via $|\tanh(g)|$) are regulated to keep them in stable ranges.

### 3. Hierarchical Combination
   * The optimizer merges global and local control:
     $$h_t = 0.3h_{\text{global}} + 0.7h_{\text{local}}, \quad e_t = 0.3e_{\text{global}} + 0.7e_{\text{local}}$$
   * This gives each parameter local adaptivity while maintaining mild global consistency — crucial for heterogeneous architectures (e.g., LLMs with both FFN and linear attention blocks).

### 4. Final Update
$$\theta_{t+1} = \theta_t - \eta \cdot h_t \, e_t \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + 0.05 F_t + 0.01 \tilde{g}_t \right)$$

Where:
* $F_t$: multi-scale memory (fractional gradient trace)
* $\tilde{g}_t$: polarity-aligned gradient
* $h_t, e_t$: hierarchical homeostatic gains
* Weight decay is decoupled (AdamW-style)

---

## 🧠 Visual Intuition

```
Gradients → tanh → |·| → mean = global coherence
      ↓
   Homeostasis feedback  ← targets ~0.8 activation
      ↓
Local updates → multi-scale memory → directional modulation
      ↓
Hierarchical merge (30% global, 70% local)
      ↓
Final adaptive step
```

---

## ⚙️ Pseudocode (Simplified)

```python
for each param group:
    g_cat = concat(all grads)
    global_energy = mean(g_cat**2)
    global_coherence = mean(abs(tanh(g_cat)))

    # weak global homeostasis (0.5× weaker, blended, clamped)
    h_g = 1 - 0.5*ρ*tanh(global_coherence - c_target)
    e_g_raw = 1 + 0.5*λ*(E_target - global_energy)
    e_g = clip(0.9*e_g_raw + 0.1, [0.925, 1.075])

    for each parameter p:
        m = β1*m + (1-β1)*g
        v = β2*v + (1-β2)*g²

        # multi-scale memory
        ema_i = ρ_i*ema_i + (1-ρ_i)*g
        energy_flow = Σ(w_i * ema_i)

        # coherence modulation
        polarity = 0.5*sign(g)*tanh(cos(m, g))
        adaptive_grad = g*(1+polarity)
        if importance(m, v) < threshold:
            adaptive_grad += ascent_strength*g

        # local homeostasis (full strength, blended, clamped)
        h_l = 1 - ρ*tanh(local_coherence - c_target)
        e_l_raw = 1 + λ*(E_target - local_energy)
        e_l = clip(0.8*e_l_raw + 0.2, [0.85, 1.15])

        # hierarchical blend
        h = 0.3*h_g + 0.7*h_l
        e = 0.3*e_g + 0.7*e_l

        # final update
        Δθ = -lr * h * e * (m/√v + 0.05*energy_flow + 0.01*adaptive_grad)
        θ ← θ * (1 - lr*wd) + Δθ
```

---

## 📊 Benchmark Summary

| Task         | Model       | Metric    | AdamW  | BiostatisV6 | Δ      | Time Overhead |
| ------------ | ----------- | --------- | ------ | ----------- | ------ | ------------- |
| CIFAR-10     | ResNet18    | Acc       | 69.23% | **72.01%**  | +2.78% | +67%          |
| CIFAR-100    | ResNet18    | Acc       | 49.78% | **52.06%**  | +2.28% | +60%          |
| Shakespeare  | GPT-1       | Train PPL | 17.39  | **13.27**   | -23.7% | +23%             |
| WikiText-103 | GPT-2 (25M) | Val PPL   | 249.68 | 248.21      | +0.59% | +34%          |

---

## 🔍 Representation Quality (CIFAR-10)

| Optimizer   | Top-5 Singular Value Concentration | Effective Rank |
| ----------- | ---------------------------------- | -------------- |
| AdamW       | 0.555                              | 9.83           |
| BiostatisV6 | **0.596**                          | **9.41**       |

➡️ Higher concentration & lower rank → more compact, structured representations.

---

## When to Use BiostatisV6

✅ **Recommended For:**

* Vision tasks (CNNs, ResNets)
* Small or noisy language models (GPT-1, Shakespeare)
* Architectures mixing different blocks (e.g. FFN + linear attention)
* Non-convex or noisy optimization landscapes

⚠️ **Use With Caution:**

* Mid-scale clean transformers (25M–100M params)
* Compute-limited setups (≈30–70% slower than AdamW)

❌ **Not Ideal:**

* Extremely stable datasets or latency-critical systems

---

## 📦 Usage

```python
from biostatis import BiostatisV6

optimizer = BiostatisV6(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    coherence_target=0.8,
    energy_target=1e-3
)
```

---

## ⚖️ Limitations

* ~1.9× optimizer memory (multi-scale EMAs)
* Slower iteration throughput (≈0.6× AdamW)
* Sparse gradients not yet supported

---

## 🔬 Research Summary

> *“BiostatisV6 stabilizes gradient dynamics via hierarchical homeostasis.
> It consistently improves vision tasks (+2–3%) and small LMs (+24%),
> while maintaining parity on well-behaved large models (GPT-2).
> This project was developed as a student-driven exploration into optimizer design, following theoretical norms as closely as possible. Any constructive feedbacks are always welcomed.”*

Future work:

* Component-wise energy targets (per module)
* Dynamic α_global scheduling
* Broader transformer-scale tuning

---

## 📚 Citation

```bibtex
@misc{choudhury2025biostatisv6,
  author       = {Hindol Roy Choudhury and Chunlin Huang},
  title        = {BiostatisV6: A Biologically-Inspired Optimizer with Hierarchical Homeostasis},
  year         = {2025},
  note         = {Student-led optimizer research project},
  url          = {https://github.com/hindolroychoudhury/BiostatisV6},
}
```
