import torch
from torch.nn import functional as F
import torch.nn as nn
import re
import tiktoken
import math

#hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16 #how many independent sequences will we process in parallel
context_size = 32 #maximum context length for predictions
max_iters = 20000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 300
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# --- Tokenizer ---
"""
Tokenizer Setup using TikToken (GPT-2 BPE)

This section initializes the tokenizer and prepares the vocabulary for the language model.

- Uses TikToken's GPT-2 Byte-Pair Encoding (BPE) tokenizer.
- Provides a fast, subword-level tokenization strategy.
- Ensures compatibility with GPT-style token indices (n_vocab = 50257).
- Defines `encoder` and `decoder` functions to convert between raw text and token ids.
- Loads the training corpus from `input.txt`.

Variables:
    enc (tiktoken.Encoding): GPT-2 BPE tokenizer.
    vocab_size (int): Size of tokenizer vocabulary.
    encoder (Callable): Function to encode raw text into token ids.
    decoder (Callable): Function to decode token ids back into text.
    text (str): Entire input corpus read from file.
"""

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
encoder = lambda s: enc.encode(s)
decoder = lambda l: enc.decode(l)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#all the unique words occurs in this text
# words = re.findall(r"\b\w+\b|[^\w\s]", text)
# vocab_size = len(words)
# vocab = sorted(set(words)) #building vocab from words
# vocab.append('<unk>')
# word_to_num_index = {w:i for i, w in enumerate(vocab)}#converting the words to numbers having index
# index_num_to_word = {i:w for i, w in enumerate(vocab)} #vice-versa


# #safe encoder with fallback to <unk>
# encoder = lambda e: [word_to_num_index.get(w, word_to_num_index['<unk>']) for w in re.findall(r"\b\w+\b|[^\w\s]", e.lower())]

# #the decoder
# decoder = lambda d: ' '.join([index_num_to_word[i] for i in d])


# --- Train/Test Split ---
"""
Data Splitting for Language Modeling

- Encodes the entire text corpus into token IDs using the GPT-2 BPE tokenizer.
- Converts the encoded token list into a PyTorch tensor of type `torch.long`.
- Splits the data into training and validation sets (90% / 10%).

Variables:
    data (Tensor): Full dataset as a 1D tensor of token IDs.
    train_data (Tensor): First 90% of `data` used for training.
    val_data (Tensor): Remaining 10% of `data` used for validation.
"""
data = torch.tensor(encoder(text), dtype=torch.long) #encoding the whole dataset
n = int(0.9*len(data)) #first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# --- Data Loader ---
"""
get_batch(split)

Generates a mini-batch of input (x) and target (y) sequences for training or validation.

Each batch consists of `batch_size` sequences of length `context_size`.

Args:
    split (str): One of 'train' or 'val' to determine the dataset source.

Returns:
    x (Tensor): Input tensor of shape (batch_size, context_size), representing context tokens.
    y (Tensor): Target tensor of shape (batch_size, context_size), representing the next tokens to predict.
"""
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    Estimates the average loss on both training and validation sets.

    - Uses `eval_iters` iterations per split to compute a stable loss estimate.
    - Runs in evaluation mode with gradient computation disabled (`@torch.no_grad`).
    - Returns a dictionary with the mean training and validation losses.

    Returns:
        out (dict): {
            'train': Mean training loss over `eval_iters` batches,
            'val': Mean validation loss over `eval_iters` batches
        }
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def compute_perplexity(model, split="val"):
    """
    Corpus-level perplexity computation that matches the training loss.
    
    Computes perplexity across all tokens in the dataset by aggregating
    log-likelihoods first, then computing exp(-mean_log_likelihood).
    
    This matches exactly with the loss computation in the model's forward pass.
    
    Args:
        model: The language model to evaluate.
        split (str): 'train' or 'val' to specify which dataset to use.
    
    Returns:
        float: The perplexity value for the specified split.
    """
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0

    for _ in range(eval_iters):
        x, y = get_batch(split)
        
        # Forward pass - returns flattened logits (B*T, vocab_size)
        logits, _ = model(x, y)
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Flatten targets to match logits shape
        targets = y.view(-1)
        
        # Gather log probabilities for the true tokens
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Accumulate
        total_log_likelihood += target_log_probs.sum().item()
        total_tokens += targets.numel()

    # Compute perplexity
    mean_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-mean_log_likelihood)

    model.train()
    return perplexity


@torch.no_grad()
def compute_perplexity_per_sequence(model, split="val"):
    """
    Per-sequence perplexity computation (article-style approach).
    
    Computes perplexity for each sequence independently, then takes the mean.
    This is better for variable-length sequences with padding/masking.
    
    Formula: mean(exp(-sum(log_probs_per_seq) / tokens_per_seq))
    
    Args:
        model: The language model to evaluate.
        split (str): 'train' or 'val' to specify which dataset to use.
    
    Returns:
        float: The mean perplexity across all sequences.
    """
    model.eval()
    all_perplexities = []

    for _ in range(eval_iters):
        x, y = get_batch(split)
        B, T = y.shape
        
        # Forward pass
        logits, _ = model(x, y)
        logits = logits.view(B, T, -1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather target log probs
        target_log_probs = log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        
        # Per-sequence negative log likelihood
        nll_per_seq = -target_log_probs.sum(dim=-1) / T  # Average per sequence
        
        # Per-sequence perplexity
        perplexities = torch.exp(nll_per_seq)
        all_perplexities.append(perplexities)

    # Mean across all sequences
    mean_perplexity = torch.cat(all_perplexities).mean().item()

    model.train()
    return mean_perplexity


def evaluate_model(model):
    """
    Runs comprehensive evaluation using both perplexity computation methods.
    
    Computes:
    1. Corpus-level perplexity (aggregates all tokens first)
    2. Per-sequence perplexity (averages individual sequence perplexities)
    
    Both for training and validation splits.
    
    Args:
        model: The language model to evaluate.
    
    Returns:
        dict: Dictionary containing all perplexity metrics:
            - 'train_ppl': Corpus-level training perplexity
            - 'val_ppl': Corpus-level validation perplexity
            - 'train_ppl_seq': Per-sequence training perplexity
            - 'val_ppl_seq': Per-sequence validation perplexity
    """
    # Corpus-level perplexity
    train_ppl = compute_perplexity(model, split="train")
    val_ppl = compute_perplexity(model, split="val")
    
    # Per-sequence perplexity
    train_ppl_seq = compute_perplexity_per_sequence(model, split="train")
    val_ppl_seq = compute_perplexity_per_sequence(model, split="val")

    print("\n" + "="*60)
    print("PERPLEXITY EVALUATION")
    print("="*60)
    print("Corpus-Level Perplexity:")
    print(f"  Train: {train_ppl:.4f}")
    print(f"  Val:   {val_ppl:.4f}")
    print("\nPer-Sequence Perplexity (Article Method):")
    print(f"  Train: {train_ppl_seq:.4f}")
    print(f"  Val:   {val_ppl_seq:.4f}")
    print("="*60 + "\n")

    return {
        "train_ppl": train_ppl, 
        "val_ppl": val_ppl,
        "train_ppl_seq": train_ppl_seq,
        "val_ppl_seq": val_ppl_seq
    }


class Head(nn.Module):
    """
    Single Head of Causal Self-Attention.

    This module computes attention scores using the query, key, and value 
    projections of the input, masks out future positions (causal masking), 
    and performs a weighted aggregation of the values.

    Args:
        head_size (int): The dimensionality of the query/key/value vectors for this head.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        """
        Forward pass for a single attention head.

        Args:
            x (Tensor): Input tensor of shape (B, T, C), where
                        B = batch size,
                        T = sequence length(context size),
                        C = embedding dimension(feature_dimension).

        Returns:
            Tensor: Output tensor of shape (B, T, head_size) after applying
                    masked scaled dot-product attention.
        """
        B,T,C = x.shape #batch,time,channel basically batch, context size, feature dimersion
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        # computing the attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 #B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Layer

    This module runs multiple self-attention heads in parallel,
    allowing the model to attend to different aspects of the input
    simultaneously (e.g., short-term, long-term, syntactic, semantic).

    It then concatenates their outputs and projects them back into the
    original embedding space to be used downstream.

    Args:
        num_heads (int): Number of parallel attention heads.
        head_size (int): Dimensionality of each individual head's key/query/value vectors.

    Input Shape:
        x: (B, T, n_embd)

    Output Shape:
        (B, T, n_embd)
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Linear projection to bring concatenated output back to embedding dimension
        self.proj = nn.Linear(num_heads * head_size, n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Each head attends to the input independently, producing different views 
        of the sequence. The outputs from all heads are concatenated and 
        projected back into the original embedding space.

        Args:
            x (Tensor): Input tensor of shape (B, T, n_embd), where
                        B = batch size,
                        T = sequence length,
                        n_embd = embedding dimension.

        Returns:
            Tensor: Output tensor of shape (B, T, n_embd), same as input shape,
                    but enriched with attention-based representations.
        """
        # Run all attention heads in parallel and concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)

        # Project back to the original embedding dimension and apply dropout
        out = self.dropout(self.proj(out))  # (B, T, n_embd)

        return out

class FeedForwardNet(nn.Module):
    """
    Feedforward Neural Network (Position-wise MLP)

    This module implements the feedforward component of a Transformer block.
    It consists of two linear transformations with a ReLU activation in between,
    followed by dropout for regularization.

    The first linear layer expands the dimensionality from n_embd to 4 * n_embd,
    enabling the model to capture more complex patterns. The second layer projects
    it back to the original embedding size to maintain consistent dimensions.

    Args:
        n_embd (int): The dimensionality of the input and output embeddings.
    """
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        """
        Forward pass of the feedforward network.

        Applies a two-layer MLP with ReLU activation in between. The input is first
        projected to a higher-dimensional space (4x the embedding size), non-linearly
        transformed, then projected back to the original embedding size. A dropout 
        is applied at the end for regularization.

        Args:
            x (Tensor): Input tensor of shape (B, T, n_embd), where
                        B = batch size,
                        T = sequence length(context size),
                        n_embd = embedding dimension.

        Returns:
            Tensor: Output tensor of shape (B, T, n_embd).
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block: Communication (Self-Attention) + Computation (Feedforward)

    This block forms the basic unit of a Transformer model. It contains two main components:
    1. A Multi-Head Self-Attention mechanism that allows the model to attend to different parts 
       of the sequence (communication).
    2. A Feedforward Neural Network (position-wise MLP) for nonlinear transformation (computation).

    Both components are preceded by Layer Normalization and employ residual connections 
    for stable training and better gradient flow.

    Args:
        n_embd (int): The dimensionality of the token embeddings.
        n_head (int): The number of attention heads to use in the multi-head attention mechanism.
    """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #multi-head self-attention layer
        self.ffn = FeedForwardNet(n_embd) #feedforward network
        self.ln1 = nn.LayerNorm(n_embd) #layer normalization before self-attention
        self.ln2 = nn.LayerNorm(n_embd) #layer normalization before feedforward network

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Applies LayerNorm → Multi-head Self-Attention → Residual Add,
        then LayerNorm → FeedForward → Residual Add.

        This pattern (Norm → SubLayer → Residual) is known as "Pre-Norm" and 
        helps with training stability.

        Args:
            x (Tensor): Input tensor of shape (B, T, n_embd), where
                        B = batch size,
                        T = sequence length,
                        n_embd = embedding dimension.

        Returns:
            Tensor: Output tensor of shape (B, T, n_embd).
        """
        x = x + self.sa(self.ln1(x)) #communication (multihead-attention)
        x = x + self.ffn(self.ln2(x)) #computation (feedforward-network)
        return x

class BigramLanguageModel(nn.Module):
    """
    Transformer-based Bigram Language Model

    This model learns to predict the next token in a sequence given a context of previous tokens.
    It is built using:
    - Token and positional embeddings
    - A stack of Transformer blocks (self-attention + feedforward)
    - A linear head to project hidden states into vocabulary logits

    While it's called a "bigram" model, this implementation uses a full transformer stack
    and learns from longer context windows (`context_size` tokens).

    Architecture:
        Input -> [Token + Positional Embedding] -> [Transformer Blocks] -> [LayerNorm] -> [Linear Head] -> Output logits

    Attributes:
        token_embedding (nn.Embedding): Maps token indices to embedding vectors.
        position_embedding_table (nn.Embedding): Provides positional encodings to preserve token order.
        blocks (nn.Sequential): Stack of `Block` layers (Transformer blocks).
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Maps final hidden state to vocabulary logits.
    """
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.

        Args:
            idx (Tensor): Input tensor of token indices with shape (B, T)
                          B = batch size, T = context window length.
            targets (Tensor, optional): Ground-truth next token indices for loss computation.
                                        Shape should be (B, T). If None, loss is not computed.

        Returns:
            logits (Tensor): Raw prediction scores for the next token (B, T, vocab_size)
            loss (Tensor or None): Cross-entropy loss between predictions and targets, if targets is provided.
        """
        B,T = idx.shape

        #idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively from the model.

        At each step, the model:
        - Conditions on the last `context_size` tokens
        - Predicts the next token
        - Samples from the logits with optional temperature scaling and top-k filtering

        Args:
            idx (Tensor): Initial sequence of token indices of shape (B, T).
            max_new_tokens (int): Number of tokens to generate.
            temperature (float): Softens the logits; lower = more confident, higher = more random.
            top_k (int or None): If set, limits sampling to top-k most probable tokens.

        Returns:
            Tensor: Sequence of token indices of shape (B, T + max_new_tokens).
        """
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
    


# --- Train the model ---

print("="*60)
print("GPT-1 Training with AdamW Optimizer")
print("="*60)

model = BigramLanguageModel().to(device)

# Print total number of model parameters in millions
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nOptimizer: AdamW")
print(f"Learning rate: {learning_rate}")
print(f"Training iterations: {max_iters}")
print(f"Batch size: {batch_size}, Context size: {context_size}\n")

# ========================
# Main Training Loop
# ========================
"""
Trains the language model over `max_iters` iterations.

At each iteration:
- A batch of input and target sequences is sampled from training data
- Forward pass computes predictions and loss
- Backpropagation updates model weights using AdamW optimizer

Every `eval_interval` steps:
- Estimates average train and validation loss over `eval_iters` batches
- Computes perplexity for both training and validation sets
- Prints the current step with corresponding losses and perplexities
"""
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = compute_perplexity(model, split="train")
        val_ppl = compute_perplexity(model, split="val")
        print(f"step {iter:5d}: "
              f"train loss {losses['train']:.4f} (ppl={train_ppl:.2f}), "
              f"val loss {losses['val']:.4f} (ppl={val_ppl:.2f})")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Final comprehensive evaluation
final_metrics = evaluate_model(model)

# --- Generate output ---
"""
Generates text from the trained model using autoregressive sampling.

Steps:
- Starts with a single zero-token (`context`)
- At each step, predicts the next token using the model
- Applies temperature and top-k sampling to increase generation quality
- Decodes the final token sequence into human-readable text
"""

print("\nGenerating sample text...\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start token (B=1, T=1)
generated_ids = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
generated_text = decoder(generated_ids[0].tolist())

print("Generated Text:")
print("-" * 60)
print(generated_text)
print("-" * 60)

# Save model with comprehensive metrics
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_train_loss': losses['train'],
    'final_val_loss': losses['val'],
    'final_metrics': final_metrics
}, 'gpt1_adamw.pth')

print("\nModel saved as 'gpt1_adamw.pth'")