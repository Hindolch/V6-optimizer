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
# BIOSTATISV6 OPTIMIZER
# ========================================
class BiostatisV6(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999),
                eps=1e-8, weight_decay=1e-2, homeo_rate=0.03, coherence_target=0.8,
                energy_target=1e-3, lambda_energy=0.1,
                memory_decays=(0.9, 0.99), memory_weights=(0.6, 0.4),
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

            # global statistics (homeostatis sensors)
            grads = [p.grad.view(-1) for p in group['params'] if p.grad is not None]
            if len(grads) == 0:
                continue
            g_cat = torch.cat(grads)
            energy = torch.mean(g_cat**2).item()
            coherence = torch.mean(torch.abs(torch.tanh(g_cat))).item()

            # homeostatic modulation
            coherence_error = coherence - coherence_target
            homeo_mod = 1.0 - homeo_rate * torch.tanh(torch.tensor(coherence_error)).item()

            # energy feedback
            raw_energy_feedback = 1 + lambda_energy * (energy_target - energy)
            energy_feedback = 0.8 * raw_energy_feedback + 0.2
            energy_feedback = max(0.85, min(1.15, energy_feedback))

            # Parameter updates
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("BiostatisV5_1_Lite does not support sparse gradients")

                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Only 2 memory channels (saves 33% memory)
                    state['memory_emas'] = [torch.zeros_like(p) for _ in memory_decays]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                memory_emas = state['memory_emas']
                state['step'] += 1

                # Adam momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Multi-scale memory (2 channels)
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

                # Final update (no layer scaling, like V5.1)
                adaptive_update = -step_size * homeo_mod * energy_feedback * (
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
def compute_perplexity(model, split="val"):
    """
    Proper token-level perplexity computation that works with your model
    which returns flattened logits of shape (B*T, vocab_size).
    """
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0

    for _ in range(eval_iters):
        x, y = get_batch(split)
        # forward pass
        logits, _ = model(x, y)

        # Reshape logits back to (B, T, vocab_size)
        B, T = y.shape
        logits = logits.view(B, T, -1)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = y[:, 1:]

        # Compute log-probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Accumulate total log likelihood
        total_log_likelihood += target_log_probs.sum().item()
        total_tokens += shift_labels.numel()

    mean_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-mean_log_likelihood)

    model.train()
    return perplexity



def evaluate_model(model):
    """
    Runs evaluation for both train and val splits.
    Returns: dict with {'train_ppl': ..., 'val_ppl': ...}
    """
    train_ppl = compute_perplexity(model, split="train")
    val_ppl = compute_perplexity(model, split="val")

    print("\n==============================")
    print("PERPLEXITY EVALUATION (Token-Level)")
    print("==============================")
    print(f"Train Perplexity: {train_ppl:.4f}")
    print(f"Val   Perplexity: {val_ppl:.4f}")
    print("==============================\n")

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
print("GPT-1 Training with BiostatisV6 Optimizer")
print("="*60)

model = BigramLanguageModel().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Initialize BiostatisV6 optimizer
optimizer = BiostatisV6(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.05,
    homeo_rate=0.03,
    coherence_target=0.85,
    energy_target=0.0005,
    lambda_energy=0.5,
    flip_threshold=0.15,
    ascent_strength=0.05
)

print(f"\nOptimizer: BiostatisV6")
print(f"Learning rate: {learning_rate}")
print(f"Training iterations: {max_iters}")
print(f"Batch size: {batch_size}, Context size: {context_size}\n")

# Training loop
for iter in range(max_iters):
    train_ppls, val_ppls = [], []
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = compute_perplexity(model, split="train")
        val_ppl = compute_perplexity(model, split="val")
        print(f"step {iter:5d}: "
              f"train loss {losses['train']:.4f} (ppl={train_ppl:.2f}), "
              f"val loss {losses['val']:.4f} (ppl={val_ppl:.2f})")

        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

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
