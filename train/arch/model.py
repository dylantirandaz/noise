"""Track 3's model: the Dust GPT at 12 blocks of width 1024 (16 heads of 64), a value embedding on every block, and
hashed bigram, trigram and skip-bigram tables (2^20 rows x 128 each) summed and projected into every block's value
embedding. Two continuation tables (4,097 rows x 128 each) join that sum: for each position, the token that followed
the last earlier occurrence of its token, and of its last two tokens, in the same sequence, so a block can read what
came next the last time this context appeared (615.0M parameters). GPTConfig()'s defaults are this architecture:
utils/eval.py builds it that way.
"""
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

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
    # per block into the value embedding (blocks with value embeddings only). 0 rows: off. Continuation codes
    # (CONTEXT_CODES) add a table indexed by a token that followed the suffix earlier in this sequence.
    ngram_rows: int = 1048576
    ngram_dim: int = 128
    ngram_orders: str = '23sab'
    # Copy route: for each context code listed, a learned amount (one scalar per reliability bucket of the code, from
    # zero) is added to the raw logit of the code's continuation token. '' is off.
    copy_sources: str = 'ab'


NGRAM_SKIPS = {'s': (2,), 't': (3,), 'u': (2, 3)}  # skip-gram codes: the token with the tokens that many back
HASH_MULTIPLIER = 2654435761  # Knuth's multiplicative hash constant


class ContextCode(NamedTuple):
    """One way to look a position's suffix up earlier in its own sequence and name the token that followed it."""
    suffix: int  # tokens in the suffix that must match
    kind: str  # which earlier occurrence is read, and how its reliability is bucketed (below)
    buckets: int  # reliability buckets; the code's table has one row per (bucket, token), so nothing is hashed


# kind      codes   continuation token                                   bucket
# 'latest'  a b c   after the most recent earlier occurrence             one
# 'agree'   A B C   the same token                                       the one before it: none, agrees, differs
# 'second'  x y z   after the second most recent earlier occurrence      one
# 'longest' m       after the latest occurrence of the longest match     the length of that match
LONGEST_SUFFIX = 5
AGREE_BUCKETS = 3
CONTEXT_CODES = {
    'a': ContextCode(1, 'latest', 1), 'b': ContextCode(2, 'latest', 1), 'c': ContextCode(3, 'latest', 1),
    'A': ContextCode(1, 'agree', AGREE_BUCKETS), 'B': ContextCode(2, 'agree', AGREE_BUCKETS),
    'C': ContextCode(3, 'agree', AGREE_BUCKETS),
    'x': ContextCode(1, 'second', 1), 'y': ContextCode(2, 'second', 1), 'z': ContextCode(3, 'second', 1),
    'm': ContextCode(LONGEST_SUFFIX, 'longest', LONGEST_SUFFIX),
}


def suffix_ids(idx, lags, pad):
    """One integer per position for the token and the tokens at `lags` before it; `pad` before the start."""
    ids = idx
    for k in lags:
        ids = ids * (pad + 1) + F.pad(idx[:, :-k], (k, 0), value=pad)
    return ids


def context_continuations(idx, n, pad, depth=1):
    """For each of the `depth` most recent earlier occurrences of each position's last n tokens in its own sequence,
    the token that followed it; `pad` where there is no such occurrence. Causal: an earlier occurrence ends before
    the position, so the token after it is the position's own or an earlier one."""
    ids = suffix_ids(idx, range(1, n), pad)
    order = ids.argsort(dim=1, stable=True)  # by suffix, then by position
    ordered = ids.gather(1, order)
    found = []
    for k in range(1, depth + 1):
        repeats = ordered[:, k:] == ordered[:, :-k]
        earlier = F.pad(torch.where(repeats, order[:, :-k], -1), (k, 0), value=-1)
        previous = torch.empty_like(order).scatter_(1, order, earlier)
        following = idx.gather(1, (previous + 1).clamp(min=0))
        found.append(torch.where(previous >= 0, following, pad))
    return found


def context_continuation(idx, n, pad):
    """The token that followed the most recent earlier occurrence of each position's last n tokens (`pad`: none)."""
    return context_continuations(idx, n, pad)[0]


def longest_continuation(idx, longest, pad):
    """(token, bucket): the token after the most recent occurrence of the longest matching suffix of up to `longest`
    tokens, and that suffix's length less one."""
    token, bucket = torch.full_like(idx, pad), torch.zeros_like(idx)
    for length in range(1, longest + 1):  # longer suffixes overwrite shorter ones
        found = context_continuation(idx, length, pad)
        matched = found != pad
        token, bucket = torch.where(matched, found, token), torch.where(matched, length - 1, bucket)
    return token, bucket


def context_lookup(idx, code, pad):
    """(token, bucket) of a context code at each position: the continuation token it names (`pad`: none, bucket 0)
    and that token's reliability bucket."""
    suffix, kind, _ = CONTEXT_CODES[code]
    if kind == 'longest':
        return longest_continuation(idx, suffix, pad)
    if kind == 'latest':
        token = context_continuation(idx, suffix, pad)
        return token, torch.zeros_like(token)
    latest, before = context_continuations(idx, suffix, pad, depth=2)
    if kind == 'second':
        return before, torch.zeros_like(before)
    no_earlier, agrees, differs = range(AGREE_BUCKETS)
    return latest, torch.where(before == pad, no_earlier, torch.where(before == latest, agrees, differs))


def context_rows(code, pad):
    """Rows of a context code's table: per bucket, one for every token and one for "no earlier occurrence"."""
    return CONTEXT_CODES[code].buckets * (pad + 1)


def ngram_index(idx, code, rows, pad, lookups=None):
    """Row of each position's n-gram: code '2' .. '9' the token and the code - 1 before it, a skip code (NGRAM_SKIPS)
    the token and the tokens at those lags; `pad` before the start. A context code (CONTEXT_CODES): the row of its
    (bucket, continuation token), from `lookups` when the caller has them."""
    if code in CONTEXT_CODES:
        token, bucket = lookups[code] if lookups else context_lookup(idx, code, pad)
        return bucket * (pad + 1) + token
    lags = NGRAM_SKIPS.get(code) or range(1, int(code))
    h = suffix_ids(idx, lags, pad) * HASH_MULTIPLIER  # wraps in int64 for long n-grams: still a fixed hash
    return (h ^ (h >> 29)).remainder(rows)


def copy_slots(sources):
    """First copy_gate entry of each copy source (its buckets follow it), and the number of entries in all."""
    slots, used = {}, 0
    for code in sources:
        slots[code] = used
        used += CONTEXT_CODES[code].buckets
    return slots, used


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
                n: nn.Embedding(context_rows(n, padded_vocab) if n in CONTEXT_CODES else config.ngram_rows,
                                config.ngram_dim)
                for n in config.ngram_orders})
            self.ngram_proj = nn.ModuleDict({
                i: nn.Linear(config.ngram_dim, kv_dim, bias=False) for i in self.value_embeds})
        if config.copy_sources:
            self.copy_gate = nn.Parameter(torch.zeros(copy_slots(config.copy_sources)[1]))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
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
        if self.config.copy_sources:
            self.copy_gate.zero_()
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

    def context_lookups(self, idx):
        """{code: (token, bucket)} for every context code the tables or the copy route read."""
        pad = self.transformer.wte.num_embeddings
        codes = set(self.config.copy_sources) | set(self.config.ngram_orders if self.config.ngram_rows else '')
        return {code: context_lookup(idx, code, pad) for code in sorted(codes) if code in CONTEXT_CODES}

    def copy_offset(self, lookups):
        """The copy route's addition to the raw logits, [batch, token, vocab]: each source's gate for the token's
        bucket, at the logit of the continuation token it names."""
        vocab, pad = self.config.vocab_size, self.transformer.wte.num_embeddings
        sources = self.config.copy_sources
        slots, _ = copy_slots(sources)
        gates = self.copy_gate.float()
        any_token, _ = lookups[sources[0]]
        offset = torch.zeros(*any_token.shape, vocab, device=any_token.device)
        for code in sources:
            token, bucket = lookups[code]
            amount = gates[slots[code] + bucket] * (token != pad)
            offset = offset.scatter_add(2, token.clamp(max=vocab - 1).unsqueeze(-1), amount.unsqueeze(-1))
        return offset

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        x = norm(self.transformer.wte(idx))
        x0 = x
        lookups = self.context_lookups(idx)
        if self.config.ngram_rows:
            pad = self.transformer.wte.num_embeddings
            ngram = sum(table(ngram_index(idx, n, self.config.ngram_rows, pad, lookups))
                        for n, table in self.ngram_embeds.items())
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ve is not None and self.config.ngram_rows:
                ve = ve + self.ngram_proj[str(i)](ngram)
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if self.config.copy_sources:
            logits = logits + self.copy_offset(lookups)
        logits = 15 * torch.tanh(logits / 15)  # soft cap
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
