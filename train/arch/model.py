"""Track 3's model: the Dust GPT at 12 blocks of width 1024 (16 heads of 64), a value embedding on every block, and
hashed bigram, trigram and skip-bigram tables (2^20 rows x 128 each) summed and projected into every block's value
embedding (613.9M parameters). A calibrated causal cache mixes the neural prediction with continuations from the
current input prefix. GPTConfig()'s defaults are this architecture: utils/eval.py builds it that way.
"""
from dataclasses import dataclass
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex = torch.compile(flex_attention)


@cache
def window_mask(window, length, device):
    return create_block_mask(
        lambda b, h, q, k: (q >= k) & (q - k <= window), None, None, length, length, device=device,
    )


def attention(q, k, v, window):
    """Causal attention with a left window; tensors have shape [batch, token, head, dim].

    A query at position s sees keys at s - window to s. Flex attention skips the blocks outside the
    window, several times faster than a dense masked kernel on the half-window blocks.
    """
    mask = window_mask(window, q.shape[1], q.device)
    return flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask).transpose(1, 2)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 12
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 1024
    # Per block, repeating: S = half-sequence window, L = full. The last block is always L.
    window_pattern: str = 'SSSL'
    # Hashed n-gram value embeddings: per order n in ngram_orders a table of ngram_rows rows of width ngram_dim,
    # indexed by a hash of the token and the n - 1 before it (or skip-grams: NGRAM_SKIPS); their sum is projected
    # per block into the value embedding (blocks with value embeddings only). 0 rows: off.
    ngram_rows: int = 1048576
    ngram_dim: int = 128
    ngram_orders: str = '23s'


NGRAM_SKIPS = {'s': (2,), 't': (3,), 'u': (2, 3)}  # skip-gram codes: the token with the tokens that many back


def ngram_index(idx, code, rows, pad):
    """Row of each position's n-gram: code '2' .. '9' the token and the code - 1 before it, a skip code (NGRAM_SKIPS)
    the token and the tokens at those lags; `pad` before the start."""
    ids = idx
    for k in NGRAM_SKIPS.get(code) or range(1, int(code)):
        ids = ids * (pad + 1) + F.pad(idx, (k, 0), value=pad)[:, :idx.size(1)]
    h = ids * 2654435761  # wraps in int64 for long n-grams: still a fixed hash
    return (h ^ (h >> 29)).remainder(rows)


CACHE_ORDER = 4
CACHE_BINS = 1 + 3 * CACHE_ORDER


def cache_matches(idx):
    """Yield a query slice, a mask over continuation tokens, their count, and a calibration bin.

    At query q the candidate token idx[v] has already been seen (v <= q). For order k its
    preceding k tokens must equal the k tokens ending at q. Use the longest matching order
    up to CACHE_ORDER, backing off to the empirical unigram distribution of idx[:q+1].
    """
    batch, length = idx.shape
    positions = torch.arange(length, device=idx.device)
    padded = F.pad(idx, (CACHE_ORDER, 0), value=-1)
    for start in range(0, length, 64):
        stop = min(start + 64, length)
        pos = slice(start, stop)
        eligible = positions[None, :] <= positions[pos, None]
        selected = eligible.expand(batch, -1, -1)
        matched = selected
        order = torch.zeros((batch, stop - start), dtype=torch.long, device=idx.device)
        for k in range(1, CACHE_ORDER + 1):
            offset = CACHE_ORDER - k
            query = padded[:, offset + start + 1:offset + stop + 1]
            source = padded[:, offset:offset + length]
            matched = matched & (positions >= k) & (query.unsqueeze(-1) == source.unsqueeze(1))
            found = matched.any(dim=-1)
            selected = torch.where(found.unsqueeze(-1), matched, selected)
            order = order + found.long()
        support = selected.sum(dim=-1)
        # Bin zero is the unigram fallback. Each positive order has bins for 1, 2–3, or 4+ matches.
        bucket = (support >= 2).long() + (support >= 4).long()
        bins = torch.where(order == 0, 0, 1 + 3 * (order - 1) + bucket)
        yield pos, selected, support, bins


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value embeddings on every block."""
    return True


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # The value embedding is scaled per head by 2 * sigmoid(gate), read from the block input's first 32
        # channels.
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = (norm(q), norm(k))
        y = attention(q, k, v, window_size[0])
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # Pre-norm block. attn.c_proj and mlp.c_proj are the "writers" into the residual stream.
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.register_buffer('cache_weights', torch.zeros(CACHE_BINS, dtype=torch.float32))
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)  # untied from the embedding
        # The stream entering block i is resid_lambdas[i] * x + x0_lambdas[i] * x0, a learned mix of the
        # previous block's output and the normalized token embedding.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        if config.ngram_rows:
            self.ngram_embeds = nn.ModuleDict({
                n: nn.Embedding(config.ngram_rows, config.ngram_dim) for n in config.ngram_orders})
            self.ngram_proj = nn.ModuleDict({
                i: nn.Linear(config.ngram_dim, kv_dim, bias=False) for i in self.value_embeds})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        self.cache_weights.zero_()
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        if self.config.ngram_rows:  # rows like the token embedding, projections from zero
            for table in self.ngram_embeds.values():
                torch.nn.init.normal_(table.weight, mean=0.0, std=1.0)
            for proj in self.ngram_proj.values():
                torch.nn.init.zeros_(proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        # Token and value embeddings are stored in bfloat16 on the GPU
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
            if self.config.ngram_rows:
                self.ngram_embeds.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def neural_logits(self, idx):
        T = idx.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        x0 = x
        if self.config.ngram_rows:
            pad = self.transformer.wte.num_embeddings
            ngram = sum(table(ngram_index(idx, n, self.config.ngram_rows, pad))
                        for n, table in self.ngram_embeds.items())
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ve is not None and self.config.ngram_rows:
                ve = ve + self.ngram_proj[str(i)](ngram)
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)  # soft cap

    def cache_observations(self, idx, logits, targets):
        """Return the two experts' target probabilities and bin per position for forward-only calibration."""
        base_probs, cache_probs, bin_ids = [], [], []
        for pos, selected, support, bins in cache_matches(idx):
            labels = targets[:, pos]
            ignored = labels == -1
            scores = logits[:, pos]
            chosen = scores.gather(-1, labels.masked_fill(ignored, 0).long().unsqueeze(-1)).squeeze(-1)
            base_prob = (chosen - scores.logsumexp(dim=-1)).exp()
            hits = (selected & (idx.unsqueeze(1) == labels.unsqueeze(-1))).sum(dim=-1)
            base_probs.append(base_prob.masked_fill(ignored, 1.0))
            cache_probs.append((hits / support).masked_fill(ignored, 1.0))
            bin_ids.append(bins)
        return torch.cat(base_probs, 1), torch.cat(cache_probs, 1), torch.cat(bin_ids, 1)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        logits = self.neural_logits(idx)
        weights = self.cache_weights.clamp(0, 1 - torch.finfo(torch.float32).eps)
        if targets is not None:
            base_prob, cache_prob, bins = self.cache_observations(idx, logits, targets)
            w = weights[bins]
            losses = -((1 - w) * base_prob + w * cache_prob).log()
            if loss_reduction == 'none':
                return losses.reshape(-1)
            if loss_reduction == 'sum':
                return losses.sum()
            if loss_reduction == 'mean':
                return losses.sum() / (targets != -1).sum()
            raise ValueError(f'Unknown loss reduction {loss_reduction!r}')

        log_probs = torch.empty_like(logits)
        token_ids = idx.long().unsqueeze(1)
        for pos, selected, support, bins in cache_matches(idx):
            w = weights[bins].unsqueeze(-1)
            counts = torch.zeros_like(logits[:, pos], dtype=torch.int32)
            counts.scatter_add_(-1, token_ids.expand_as(selected), selected.int())
            cache_prob = counts / support.unsqueeze(-1)
            probs = (1 - w) * logits[:, pos].softmax(dim=-1) + w * cache_prob
            log_probs[:, pos] = probs.log()
        return log_probs
