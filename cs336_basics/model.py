"""Neural-network primitives built from scratch for the CS336 Transformer."""

import torch
from torch import nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor


class Linear(nn.Module):
    """A bias-free linear layer: y = x @ W.T.

    The weight is stored with shape (out_features, in_features), matching
    torch.nn.Linear (and the pretrained checkpoints), so it can be loaded with
    load_state_dict without any transposing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Truncated normal: mean 0, var 2/(d_in+d_out), clipped to [-3σ, 3σ].
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(
        self, x: Float[Tensor, " batch seq d_in"]
    ) -> Float[Tensor, " batch seq d_out"]:
        return einsum(
            x, self.weight, "batch seq d_in, d_out d_in -> batch seq d_out"
        )


class Embedding(nn.Module):
    """A token embedding table: a learnable lookup of shape (vocab_size, d_model).

    forward gathers rows by token id, so an id tensor of shape (...) returns
    (..., d_model). Weight layout matches the `token_embeddings.weight` checkpoint.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Embeddings init to N(0, 1) truncated at [-3, 3] (no fan-in/out scaling).
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: x / sqrt(mean(x²) + eps) * gain.

    Normalizes over the last (d_model) axis. No mean-subtraction and no bias,
    unlike LayerNorm; the only parameter is a per-feature gain g.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Gain initialized to 1 so the layer starts as a pure normalization.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        # Upcast to fp32 so mean(x²) doesn't overflow / lose precision in bf16.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight

        return result.to(in_dtype)


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """SiLU / swish: x * sigmoid(x). Smooth, non-monotonic, self-gating."""
    return x * torch.sigmoid(x)


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """Numerically stable softmax along `dim`.

    Subtracting the per-slice max before exp shifts every logit by a constant,
    which cancels between numerator and denominator (so the result is unchanged)
    while keeping exp's argument <= 0 to avoid overflow.
    """
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, " ... vocab"],
    targets: Int[Tensor, " ..."],
) -> Float[Tensor, ""]:
    """Average cross-entropy of `targets` under the logits `inputs`.

    loss = -log softmax(x)_t = -(x_t - logsumexp(x)). We compute logsumexp with
    the max subtracted for numerical stability, and never form softmax then log
    (which would lose precision and risk log(0)). Averaged over all examples.
    """
    # logsumexp over the vocab dim, computed stably (subtract per-row max).
    m = inputs.amax(dim=-1, keepdim=True)
    logsumexp = m.squeeze(-1) + torch.log(torch.exp(inputs - m).sum(dim=-1))

    # Gather the logit of the correct class for each example.
    target_logit = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return (logsumexp - target_logit).mean()


class SwiGLU(nn.Module):
    """Gated feed-forward block: W2( SiLU(W1 x) ⊙ (W3 x) ).

    W1, W3 up-project d_model -> d_ff; W2 down-projects d_ff -> d_model.
    SiLU(W1 x) acts as a learned gate on the W3 x value path. No biases.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """softmax(QKᵀ / √d_k + mask) V.

    Scores are scaled by 1/√d_k so their variance stays ~constant as the head
    dimension grows, keeping softmax out of its saturated, low-gradient region.
    Where `mask` is False the score is set to -inf, so that key gets zero weight.
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / (d_k**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... q k, ... k d_v -> ... q d_v")


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    Rotates each consecutive pair of dimensions (2i, 2i+1) of a query/key
    vector by an angle position * theta^(-2i/d_k). Because rotations compose,
    the dot product of a rotated query and key depends only on their relative
    position, not their absolute positions.

    cos/sin for every position in [0, max_seq_len) are precomputed once and
    stored as non-persistent buffers (not saved in state_dict), then indexed
    by the per-token positions at forward time.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even head dimension"
        self.d_k = d_k

        # One frequency per dimension pair: theta^(-2i/d_k), i = 0 .. d_k/2-1.
        i = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-i / d_k)  # (d_k/2,)

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # angles[p, i] = p * inv_freq[i]  ->  (max_seq_len, d_k/2)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_k"],
        token_positions: Int[Tensor, " ... seq"],
    ) -> Float[Tensor, " ... seq d_k"]:
        # Look up the precomputed angles for these positions: (..., seq, d_k/2).
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Split each vector into its (even, odd) pairs: (..., seq, d_k/2).
        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        # 2-D rotation of every pair by its position-dependent angle.
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.stack((out_even, out_odd), dim=-1)
        return out.reshape(*x.shape)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Projects the input to Q, K, V, splits each into `num_heads` heads of size
    d_k = d_model // num_heads, runs scaled dot-product attention per head with a
    causal mask (each query attends only to itself and earlier keys), then
    concatenates the heads and applies the output projection. If a RoPE module is
    given, it rotates Q and K (per head) before attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: "RotaryPositionalEmbedding | None" = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        seq = x.shape[-2]

        # Project, then split d_model into (head, d_k) and move head before seq.
        Q = rearrange(self.q_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (h d) -> ... h seq d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            # Add a head axis so the same positions broadcast across all heads.
            pos = token_positions.unsqueeze(-2)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # Causal mask: query i may attend to key j only if j <= i.
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))

        attn = scaled_dot_product_attention(Q, K, V, mask)  # (... h seq d_k)
        attn = rearrange(attn, "... h seq d -> ... seq (h d)")
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Two residual sub-layers: y = x + Attn(RMSNorm(x)); out = y + FFN(RMSNorm(y)).
    Normalizing inside each residual branch (pre-norm) keeps the residual stream
    clean and trains more stably than post-norm. Attention uses RoPE.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: "RotaryPositionalEmbedding | None" = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Embeds token ids, runs a stack of pre-norm Transformer blocks (all sharing
    one RoPE module), applies a final RMSNorm, and projects to vocab-size logits.
    Output is unnormalized (no softmax) — the loss applies softmax itself.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.context_length = context_length
        rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self, in_indices: Int[Tensor, " ... seq"]
    ) -> Float[Tensor, " ... seq vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
