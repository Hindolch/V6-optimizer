# import torch
# import torch.distributed as dist


# def zeropower_via_newtonschulz5(G, steps: int):
#     """
#     Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
#     quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
#     of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
#     zero even beyond the point where the iteration no longer converges all the way to one everywhere
#     on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
#     where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
#     performance at all relative to UV^T, where USV^T = G is the SVD.
#     """
#     assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
#     a, b, c = (3.4445, -4.7750,  2.0315)
#     X = G.bfloat16()
#     if G.size(-2) > G.size(-1):
#         X = X.mT

#     # Ensure spectral norm is at most 1
#     X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
#     # Perform the NS iterations
#     for _ in range(steps):
#         A = X @ X.mT
#         B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
#         X = a * X + B @ X
    
#     if G.size(-2) > G.size(-1):
#         X = X.mT
#     return X


# def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
#     momentum.lerp_(grad, 1 - beta)
#     update = grad.lerp_(momentum, beta) if nesterov else momentum
#     if update.ndim == 4: # for the case of conv filters
#         update = update.view(len(update), -1)
#     update = zeropower_via_newtonschulz5(update, steps=ns_steps)
#     update *= max(1, grad.size(-2) / grad.size(-1))**0.5
#     return update


# class Muon(torch.optim.Optimizer):
#     """
#     Muon - MomentUm Orthogonalized by Newton-schulz

#     https://kellerjordan.github.io/posts/muon/

#     Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
#     processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
#     matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
#     advantage that it can be stably run in bfloat16 on the GPU.

#     Muon should only be used for hidden weight layers. The input embedding, final output layer,
#     and any internal gains or biases should be optimized using a standard method such as AdamW.
#     Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
#     collapsing their last 3 dimensions.

#     Arguments:
#         lr: The learning rate, in units of spectral norm per update.
#         weight_decay: The AdamW-style weight decay.
#         momentum: The momentum. A value of 0.95 here is usually fine.
#     """
#     def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
#         defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
#         assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
#         params = sorted(params, key=lambda x: x.size(), reverse=True)
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params = group["params"]
#             params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
#             for base_i in range(len(params))[::dist.get_world_size()]:
#                 if base_i + dist.get_rank() < len(params):
#                     p = params[base_i + dist.get_rank()]
#                     if p.grad is None:
#                         # continue
#                         p.grad = torch.zeros_like(p)  # Force synchronization
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state["momentum_buffer"] = torch.zeros_like(p)
#                     update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
#                     p.mul_(1 - group["lr"] * group["weight_decay"])
#                     p.add_(update.reshape(p.shape), alpha=-group["lr"])
#                 dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

#         return loss


# class SingleDeviceMuon(torch.optim.Optimizer):
#     """
#     Muon variant for usage in non-distributed settings.
#     """
#     def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
#         defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     # continue
#                     p.grad = torch.zeros_like(p)  # Force synchronization
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state["momentum_buffer"] = torch.zeros_like(p)
#                 update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
#                 p.mul_(1 - group["lr"] * group["weight_decay"])
#                 p.add_(update.reshape(p.shape), alpha=-group["lr"])

#         return loss


# def adam_update(grad, buf1, buf2, step, betas, eps):
#     buf1.lerp_(grad, 1 - betas[0])
#     buf2.lerp_(grad.square(), 1 - betas[1])
#     buf1c = buf1 / (1 - betas[0]**step)
#     buf2c = buf2 / (1 - betas[1]**step)
#     return buf1c / (buf2c.sqrt() + eps)


# class MuonWithAuxAdam(torch.optim.Optimizer):
#     """
#     Distributed Muon variant that can be used for all parameters in the network, since it runs an
#     internal AdamW for the parameters that are not compatible with Muon. The user must manually
#     specify which parameters shall be optimized with Muon and which with Adam by passing in a
#     list of param_groups with the `use_muon` flag set.

#     The point of this class is to allow the user to have a single optimizer in their code, rather
#     than having both a Muon and an Adam which each need to be stepped.

#     You can see an example usage below:

#     https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
#     ```
#     hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
#     embed_params = [p for n, p in model.named_parameters() if "embed" in n]
#     scalar_params = [p for p in model.parameters() if p.ndim < 2]
#     head_params = [model.lm_head.weight]

#     from muon import MuonWithAuxAdam
#     adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
#     adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
#     muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
#     param_groups = [*adam_groups, muon_group]
#     optimizer = MuonWithAuxAdam(param_groups)
#     ```
#     """
#     def __init__(self, param_groups):
#         for group in param_groups:
#             assert "use_muon" in group
#             if group["use_muon"]:
#                 group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
#                 # defaults
#                 group["lr"] = group.get("lr", 0.02)
#                 group["momentum"] = group.get("momentum", 0.95)
#                 group["weight_decay"] = group.get("weight_decay", 0)
#                 assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
#             else:
#                 # defaults
#                 group["lr"] = group.get("lr", 3e-4)
#                 group["betas"] = group.get("betas", (0.9, 0.95))
#                 group["eps"] = group.get("eps", 1e-10)
#                 group["weight_decay"] = group.get("weight_decay", 0)
#                 assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
#         super().__init__(param_groups, dict())

#     @torch.no_grad()
#     def step(self, closure=None):

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             if group["use_muon"]:
#                 params = group["params"]
#                 params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
#                 for base_i in range(len(params))[::dist.get_world_size()]:
#                     if base_i + dist.get_rank() < len(params):
#                         p = params[base_i + dist.get_rank()]
#                         if p.grad is None:
#                             # continue
#                             p.grad = torch.zeros_like(p)  # Force synchronization
#                         state = self.state[p]
#                         if len(state) == 0:
#                             state["momentum_buffer"] = torch.zeros_like(p)
#                         update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
#                         p.mul_(1 - group["lr"] * group["weight_decay"])
#                         p.add_(update.reshape(p.shape), alpha=-group["lr"])
#                     dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
#             else:
#                 for p in group["params"]:
#                     if p.grad is None:
#                         # continue
#                         p.grad = torch.zeros_like(p)  # Force synchronization
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state["exp_avg"] = torch.zeros_like(p)
#                         state["exp_avg_sq"] = torch.zeros_like(p)
#                         state["step"] = 0
#                     state["step"] += 1
#                     update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
#                                          state["step"], group["betas"], group["eps"])
#                     p.mul_(1 - group["lr"] * group["weight_decay"])
#                     p.add_(update, alpha=-group["lr"])

#         return loss


# class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
#     """
#     Non-distributed variant of MuonWithAuxAdam.
#     """
#     def __init__(self, param_groups):
#         for group in param_groups:
#             assert "use_muon" in group
#             if group["use_muon"]:
#                 # defaults
#                 group["lr"] = group.get("lr", 0.02)
#                 group["momentum"] = group.get("momentum", 0.95)
#                 group["weight_decay"] = group.get("weight_decay", 0)
#                 assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
#             else:
#                 # defaults
#                 group["lr"] = group.get("lr", 3e-4)
#                 group["betas"] = group.get("betas", (0.9, 0.95))
#                 group["eps"] = group.get("eps", 1e-10)
#                 group["weight_decay"] = group.get("weight_decay", 0)
#                 assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
#         super().__init__(param_groups, dict())

#     @torch.no_grad()
#     def step(self, closure=None):

#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             if group["use_muon"]:
#                 for p in group["params"]:
#                     if p.grad is None:
#                         # continue
#                         p.grad = torch.zeros_like(p)  # Force synchronization
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state["momentum_buffer"] = torch.zeros_like(p)
#                     update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
#                     p.mul_(1 - group["lr"] * group["weight_decay"])
#                     p.add_(update.reshape(p.shape), alpha=-group["lr"])
#             else:
#                 for p in group["params"]:
#                     if p.grad is None:
#                         # continue
#                         p.grad = torch.zeros_like(p)  # Force synchronization
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state["exp_avg"] = torch.zeros_like(p)
#                         state["exp_avg_sq"] = torch.zeros_like(p)
#                         state["step"] = 0
#                     state["step"] += 1
#                     update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
#                                          state["step"], group["betas"], group["eps"])
#                     p.mul_(1 - group["lr"] * group["weight_decay"])
#                     p.add_(update, alpha=-group["lr"])

#         return loss









import torch
from torch.nn import functional as F
import torch.nn as nn
import re
import tiktoken
import math

# ========================================
# HYPERPARAMETERS
# ========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
context_size = 32
max_iters = 20000
eval_interval = 200
learning_rate = 2e-3
eval_iters = 300
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# ========================================
# BIOSTATISV5 OPTIMIZER
# ========================================
class BiostatisV5(torch.optim.Optimizer):
    """
    Memory-optimized BiostatisV5 for GPT-1 training.
    Includes homeostatic modulation, multi-scale memory, and layer-wise scaling.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2,
                 homeo_rate=0.03, coherence_target=0.8,
                 energy_target=1e-3, lambda_energy=0.1,
                 memory_decays=(0.9, 0.95, 0.99),
                 memory_weights=(0.5, 0.3, 0.2),
                 flip_threshold=0.2, ascent_strength=0.05):
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            homeo_rate=homeo_rate, coherence_target=coherence_target,
            energy_target=energy_target, lambda_energy=lambda_energy,
            memory_decays=memory_decays, memory_weights=memory_weights,
            flip_threshold=flip_threshold, ascent_strength=ascent_strength
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            homeo_rate = group['homeo_rate']
            coherence_target = group['coherence_target']
            energy_target = group['energy_target']
            lambda_energy = group['lambda_energy']
            memory_decays = group['memory_decays']
            memory_weights = group['memory_weights']
            flip_threshold = group['flip_threshold']
            ascent_strength = group['ascent_strength']

            # Global statistics
            grads = [p.grad.view(-1) for p in group['params'] if p.grad is not None]
            if len(grads) == 0:
                continue
            g_cat = torch.cat(grads)
            energy = torch.mean(g_cat ** 2).item()
            coherence = torch.mean(torch.abs(torch.tanh(g_cat))).item()

            # Homeostatic modulation
            coherence_error = coherence - coherence_target
            homeo_mod = 1.0 - homeo_rate * torch.tanh(torch.tensor(coherence_error)).item()

            # Energy feedback
            raw_energy_feedback = 1 + lambda_energy * (energy_target - energy)
            energy_feedback = 0.8 * raw_energy_feedback + 0.2
            energy_feedback = max(0.85, min(1.15, energy_feedback))

            # Layer-wise scaling
            layer_means = []
            for p in group['params']:
                if p.grad is not None and 'exp_avg_sq' in self.state[p]:
                    layer_means.append(self.state[p]['exp_avg_sq'].mean().item())
            
            if len(layer_means) > 0:
                mean_energy = sum(layer_means) / len(layer_means)
            else:
                mean_energy = 1e-8
            layer_scale = 1.0 / ((mean_energy + 1e-8) ** 0.25)
            layer_scale = min(max(layer_scale, 0.7), 1.3)

            # Parameter updates
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("BiostatisV5 does not support sparse gradients")

                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['memory_emas'] = [torch.zeros_like(p) for _ in memory_decays]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                memory_emas = state['memory_emas']
                state['step'] += 1

                # Adam momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Multi-scale memory
                for ema, decay in zip(memory_emas, memory_decays):
                    ema.mul_(decay).add_(grad, alpha=1 - decay)
                
                energy_flow = torch.zeros_like(grad)
                for weight, ema in zip(memory_weights, memory_emas):
                    energy_flow.add_(ema, alpha=weight)

                # Coherence modulation
                flow_change = torch.cosine_similarity(exp_avg.flatten(), grad.flatten(), dim=0)
                polarity = 0.5 * torch.sign(grad) * torch.tanh(flow_change)
                adaptive_grad = grad * (1.0 + polarity)

                # Selective ascent
                importance = exp_avg.abs().mean() / (exp_avg_sq.sqrt().mean() + 1e-12)
                ascent_decay = 1 - math.exp(-0.02 * state['step'])
                if importance < flip_threshold:
                    adaptive_grad.add_(ascent_strength * ascent_decay * grad)

                # Compute step
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Final update
                adaptive_update = -step_size * homeo_mod * energy_feedback * layer_scale * (
                    exp_avg / denom + 0.05 * energy_flow + 0.01 * adaptive_grad
                )

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                p.add_(adaptive_update)

        return loss


# ========================================
# TOKENIZER
# ========================================
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encoder = lambda s: enc.encode(s)
decoder = lambda l: enc.decode(l)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ========================================
# DATA PREPARATION
# ========================================
data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
@torch.no_grad()
def compute_perplexity(model):
    model.eval()
    losses = estimate_loss()
    train_ppl = math.exp(losses['train'])
    val_ppl = math.exp(losses['val'])
    print("\n==============================")
    print("PERPLEXITY EVALUATION")
    print("==============================")
    print(f"Train Loss: {losses['train']:.4f}  →  Train Perplexity: {train_ppl:.2f}")
    print(f"Val   Loss: {losses['val']:.4f}  →  Val   Perplexity: {val_ppl:.2f}")
    print("==============================\n")
    model.train()
    return {"train_ppl": train_ppl, "val_ppl": val_ppl}

# ========================================
# MODEL ARCHITECTURE
# ========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForwardNet(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForwardNet(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
    
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
    
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ========================================
# TRAINING
# ========================================
print("="*60)
print("GPT-1 Training with BiostatisV5 Optimizer")
print("="*60)

model = BigramLanguageModel().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Initialize BiostatisV5 optimizer
optimizer = BiostatisV5(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.05,    # Higher regularization (was 1e-2)
    homeo_rate=0.02,      # More conservative (was 0.05)
    coherence_target=0.85, # Higher coherence for language
    energy_target=5e-4,   # Lower energy target
    lambda_energy=0.05,   # Weaker energy feedback
    flip_threshold=0.15,  # Less aggressive ascent
    ascent_strength=0.02  # Weaker ascent
)

print(f"\nOptimizer: BiostatisV5")
print(f"Learning rate: {learning_rate}")
print(f"Training iterations: {max_iters}")
print(f"Batch size: {batch_size}, Context size: {context_size}\n")

# Training loop

for iter in range(max_iters):
    train_ppls, val_ppls = [], []
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses['val'])
        print(f"step {iter:5d}: "
              f"train loss {losses['train']:.4f} (ppl={train_ppl:.2f}), "
              f"val loss {losses['val']:.4f} (ppl={val_ppl:.2f})")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
p = compute_perplexity(model)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# ========================================
# GENERATION
# ========================================
print("\nGenerating sample text...\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
generated_text = decoder(generated_ids[0].tolist())

print("Generated Text:")
print("-" * 60)
print(generated_text)
print("-" * 60)

# Save model
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'final_train_loss': losses['train'],
#     'final_val_loss': losses['val']
# }, 'gpt1_biostatisv5.pth')

# print("\nModel saved as 'gpt1_biostatisv5.pth'")