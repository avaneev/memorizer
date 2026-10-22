# memorizer4

A memoriser compressor that learns how far to trust its own predictions instead of
writing that trust down.

```
memorizer4 c <input> <output>    compress
memorizer4 d <input> <output>    decompress
memorizer4 t <input> [S]         size, ratio, throughput, channels, round-trip check
```

C++11, ASCII-only source, in-memory I/O, no external libraries.

## 1. The problem this design solves

Compression here means *transmit a model that reproduces the input, then repair the
positions the model gets wrong*. The model is a trie of contexts over a 4-bit symbol
stream:

- the input bytes are re-blocked into **nibbles**, so a 500 KB file becomes a million
  positions;
- every distinct context of length 1..Q (Q = 8) is a node;
- a node's **answer** is the symbol that most often follows its context.

Two reductions make the trie small enough to transmit. **Mode-preserving collapse**
drops a node whose answer equals its parent's answer — a position whose deepest match
is dropped gets the same answer from its nearest surviving ancestor, so no prediction
changes. **Support pruning** drops a node whose context occurs fewer than S = 8 times,
because a node costs about 10.4 bits to transmit and a repaired miss about 2.9.

```
kept(n)    = n survives the collapse and support(n) >= S
closure(n) = kept(n), or n lies on a path from the root to a kept node
```

A closure node that is not kept is pure structure: it carries no answer, only an
"inherit" code, and exists so the decoder can walk to a kept descendant.

**The cost of one answer per node.** A memoriser that states only an answer leaves the
decoder unable to tell whether that answer is right. The obvious repair is to
transmit, alongside the model, a per-position flag saying *right or wrong* and a
residual saying *what it should have been*. Those two channels then carry the entire
cost of the model self-assessment: one bit and one rank for every position, almost all
of it the model narrating its own mistakes one at a time.

**The cost of stating the confidence.** A better repair gives each node its own
reliability: the rate at which its answer has been correct, and a **fallback** symbol
for the cases where it has not. Then a position asks three questions — *was it the
answer*, *was it the fallback*, *which of the other fourteen* — and no flag or
residual channel is needed. But a rate is a number, and stating it spends model bytes
at every node.

What this design does instead: **it states the answer and the fallback, and learns
both rates by counting.**

## 2. The idea

Every kept node has two Bernoulli outcomes attached to it: whether its answer was
right, and — among the misses — whether the missing symbol was the fallback. Both are
properties of the model's own behaviour that the decoder can simply *observe* as it
decodes.

So it does:

```
both sides keep, per node,    hits / spans        for the answer rate
                              fbHits / fbSpans    for the fallback rate

before coding position t:     p = (hits + 1/2) / (spans + 1)
after  coding position t:     record the outcome and move on
```

Nothing about either rate is transmitted. The decoder reconstructs a node's
reliability from the positions it has already decoded at that node — the model's
confidence is memorised by the model's own traffic.

**Why this beats stating the rate.** Estimating a Bernoulli rate online from `n`
observations costs about `½·log₂(n)` bits of redundancy relative to knowing it in
advance. Stating the rate once in a small code costs 4 to 6 bits per node, because
the code has to be written for every node whether it is visited twice or a thousand
times.

The comparison is not per node but per corpus. This corpus has **27,788 kept nodes**
and **5,621,557 positions**, so a kept node is visited about **202 times** on average.
At `n = 202` the learning redundancy is `½·log₂(202) ≈ 3.8` bits. The crossover, where
stating becomes cheaper than learning, is around `n = 2⁹ = 512` visits — and most
nodes never approach it, because a node's visit count is bounded by its support and
most nodes were pruned precisely for having low support.

The two populations should be kept apart. The closure is 44,566 nodes, and the 10.4
bits a node costs to transmit is per *closure* node; the visit average above is per
*kept* node, because a structure-only node carries no rate and receives no visits of
its own — a position that reaches one either continues to a kept descendant or falls
back to its nearest ancestor, and either way contributes nothing to that node's rate.

That is the same accounting that makes support pruning pay, applied one level up: a
node is worth stating its answer for, but a node's *stated reliability* is worth less
than learning it from the node's traffic.

### 2.1 The estimator, and why it is a multiply

The estimate used is the **Krichevsky–Trofimov** one, `(hits + ½)/(spans + 1)`, which
is the natural choice for a Bernoulli source with an unknown rate: it minimises the
worst-case redundancy over all rates, and its total regret over `n` observations is
`½·log₂(n) + O(1)`.

The coder wants an 11-bit probability, so the value needed at each position is

```
p₁₁ = 2048 · (2·hits + 1) / (2·spans + 2)
```

Evaluating that literally costs an integer division per position — on the order of
twenty cycles, several million times per file. The denominator depends only on
`spans`, and `spans` is capped (see below), so the reciprocals are tabulated once:

```
recip[s] = 2³² / (2s + 2)                    for s = 0 .. COUNT_LIMIT
p₁₁      = ((2·hits + 1) · recip[spans] + 2²⁰) >> 21
```

The shift of 21 is `32 − 11`: the table stores the reciprocal in the same fixed point
the answer needs. The truncated reciprocal makes the multiply agree with the division
exactly on 99.945% of the reachable `(hits, spans)` pairs and differ by one part in
2048 on the rest. The worst single disagreement is 0.0144 bits, at the low-confidence end
where one 11-bit step is a large fraction of a small probability; at the high-confidence
end it is 0.0007 bits. The formula is identical on both sides of the coder, so no
disagreement can affect a round trip. The result
is clamped to `[1, 2047]` because a probability the coder cannot represent is worse
than a slightly wrong one. **No division appears on the coding path.**

`COUNT_LIMIT = 4095` caps the counters. Once a node has been seen that many times its
estimate has stopped moving, so both counts are halved and counting continues: the
ratio survives, the magnitude does not, and the table stays at 16 KB. The halving is
also a mild recency weighting, which is a small advantage rather than a cost.

## 3. The algorithm

### 3.1 Construction by one radix sort

```
build(x[0..M-1], Q, S):
    # 1. pack the Q-symbol key of every position
    for t in Q..M-1:
        key[t] = pack(x[t-1], x[t-2], ..., x[t-Q])      # most recent in the high nibble

    # 2. one stable radix sort of the positions by key (ceil(4Q/8) counting passes)
    idx = radix_sort(positions Q..M-1 by key)

    # 3. every depth is already grouped: equal (d)-suffixes are a contiguous run of the
    #    (d+1)-ordered array, so shifting the key finds the boundaries
    for d in Q down to 1:
        for each maximal run of idx with equal (key >> 4*(Q-d)) & mask:
            groups[d].append(run_key, run_length, argmax_symbol(x[run]))

    # 4. kept(n) = mode differs from the parent's mode AND support >= S
    for d in 1..Q:
        for each group g at depth d:
            parent = binary_search(groups[d-1], g.key >> 4)
            kept[g] = (g.mode != parent.mode) and (g.support >= S)

    # 5. closure = the trie built from the kept nodes; the builder creates the path
    #    nodes automatically, with no answer of their own
    closure = trie()
    for d in 1..Q:
        for each kept group g at depth d:
            node = descend(closure, g.key's d symbols)
            node.answer = g.mode
```

### 3.2 The only statistics that are transmitted

The fallback symbol has to be known *before* the position is coded, so it cannot be
learned from the positions themselves. It comes from one walk of the input over the
finished closure, and it is the only per-node statistic in the model:

```
fallbacks(x, closure, rootAnswer):
    for t in 1..M-1:
        n, answer = walk_to_node(closure, background of t, rootAnswer)
        if x[t] != answer:  miss[n][ rank_of(x[t], excluding answer) ] += 1
    for each node n:
        f = argmax over the 15 non-answer symbols of miss[n]
        fbSym[n] = symbol of f, or NONE          # undo the rank
```

Both rates start at zero and are learned while coding.

### 3.3 Coding a position

```
encode_positions(x, closure, rootAnswer):
    for every node: rate = {} , fbRate = {}          # both counts at zero

    for t in 1..M-1:
        n, answer = walk_to_node(...)
        fb        = fbSym[n]
        p0        = estimate(rate[n])
        p1        = estimate(fbRate[n])

        if x[t] == answer:      bit(p0, 0)
        else:
            bit(p0, 1)                      # a miss
            if fb != NONE:                  # this bit exists only with a fallback
                if x[t] == fb:  bit(p1, 0)
                else:           bit(p1, 1)
            if x[t] != fb or fb == NONE:
                tree(tail, 4, rank_of(x[t], excluding answer and fb))

        observe( rate[n],   x[t] == answer )
        if x[t] != answer and fb != NONE:
            observe( fbRate[n], x[t] == fb )

estimate(L)   = clamp( ((2*L.hit+1) * recip[L.span] + 2^20) >> 21 , 1, 2047 )
observe(L,ok) = L.hit += ok; L.span += 1;
                if L.span == COUNT_LIMIT: L.hit >>= 1; L.span >>= 1
```

Decoding is the mirror image, and the two `observe` calls are the same lines in the
same order — that identity is the whole mechanism. If the encoder and decoder updated
their counts differently the two would diverge without any checksum noticing, so the
update is deliberately the last thing in the loop body on both sides.

The rank mapping is delicate. The two excluded symbols are sorted ascending and both
tests read the *original* symbol:

```
rank = symbol - (symbol > first) - (second exists and symbol > second)
```

and the inverse tests the value produced so far, ascending:

```
symbol = rank; if (symbol >= first) symbol++; if (second exists and symbol >= second) symbol++
```

## 4. Vocabulary

- **symbol** — a 4-bit nibble; the byte stream is split into two per byte.
- **context** — the `d` symbols immediately before a position, most recent first.
- **answer / mode** — the most frequent successor of a context over the input.
- **collapse** — deleting a node whose answer equals its parent's answer.
- **kept node** — survives the collapse and meets the support threshold.
- **closure** — the kept nodes plus every node on a path to one.
- **support** — how many positions carry a given context.
- **fallback** — the symbol that most often replaced the answer among a node's misses.
  Transmitted once per node.
- **rate, fallback rate** — how often a node's answer has been right, and how often a
  miss has been its fallback. Learned, never transmitted.
- **estimate** — the Krichevsky–Trofimov value `(hits + ½)/(spans + 1)`, evaluated as
  a multiply by tabulated reciprocals.
- **tail** — the fourteen symbols that are neither the answer nor the fallback.
- **channel** — an independently charged part of the output.

## 5. Measured results

Canterbury corpus, 2,810,784 bytes, each file compressed as a standalone input.
Throughput is in memory, excluding file I/O. Sizes are exact and reproducible; the
MB/s figures move a few percent with machine load.

| file | bytes | compressed | ratio | MB/s compress | MB/s decompress |
|---|---|---|---|---|---|
| alice29.txt | 152,089 | 57,392 | 0.3774 | 4.76 | 12.60 |
| asyoulik.txt | 125,179 | 51,467 | 0.4111 | 4.60 | 11.88 |
| cp.html | 24,603 | 11,244 | 0.4570 | 3.81 | 15.25 |
| fields.c | 11,150 | 4,934 | 0.4425 | 4.33 | 17.18 |
| grammar.lsp | 3,721 | 2,009 | 0.5399 | 3.37 | 16.84 |
| kennedy.xls | 1,029,744 | 251,314 | 0.2441 | 9.06 | 34.10 |
| lcet10.txt | 426,754 | 143,155 | 0.3355 | 4.50 | 10.66 |
| plrabn12.txt | 481,861 | 181,942 | 0.3776 | 4.49 | 10.46 |
| ptt5 | 513,216 | 62,840 | 0.1224 | 10.39 | 33.41 |
| sum | 38,240 | 18,344 | 0.4797 | 3.97 | 16.88 |
| xargs.1 | 4,227 | 2,629 | 0.6220 | 2.75 | 14.18 |
| **total** | **2,810,784** | **787,270** | **0.2801** | **6.27** | **17.69** |

The support threshold is `S = 8`, swept over the corpus:

| S | 4 | 6 | **8** | 10 | 12 | 16 |
|---|---|---|---|---|---|---|
| total bytes | 798,553 | 789,442 | **787,270** | 787,329 | 788,489 | 792,750 |
| ratio | 0.2841 | 0.2809 | **0.2801** | 0.2801 | 0.2805 | 0.2820 |

The curve is flat between 8 and 10 and rises both ways. Seven of the nine text files
prefer a threshold above 6, and the smallest ones prefer the highest values: a node
that is stated costs bytes whatever the file size, while its visits — and therefore
the savings it buys — scale with the file. That is why there is no small-input special
case, and why the threshold sits above the point where a node merely "pays for itself"
on average.

All eleven files round-trip byte-exactly. Edge cases verified: empty, 1, 2 and 3
bytes, a 2,000,000-byte run of 0x00 (410 B), a 1,000,000-byte run of 0xFF (230 B), an
odd-length 100,001-byte file, and 100,000 random bytes. Two compressions of the same
input are byte-identical.

| channel | bytes | share |
|---|---|---|
| C1 model | 58,275 | 7.4% |
| C3 positions | 728,555 | 92.5% |
| header and two checksums | 440 | 0.1% |

The model is 7.4% of the output because it states only what the decoder cannot learn:
the closure's shape, its answers and its fallbacks. Everything the decoder can observe for
itself — both rates — is absent from it.

## 6. Known limits

- **The fallback symbol is transmitted.** It must be known before the position is
  coded, so it cannot be learned from the same positions. Learning it from earlier
  visits is possible but would code the early ones against a symbol that is not yet the
  right one.
- **The tail is coded adaptively and globally**, the same tree for every node. A
  per-node tail would recover more, at a table cost not paid here.
- **The encoder needs about 24 bytes per input byte** for the radix sort, so the
  512 MiB cap implies roughly 12 GiB of working set at the limit. An allocation failure
  anywhere in the model build is caught and reported rather than terminating.
- **The declared-size check has a tolerance.** A stream whose declared size is short by
  up to about 50 output bytes is accepted, because each section reader is compared
  against its length with 16 bytes of slack for the coder flush. A forged header can
  therefore change the output length slightly without rejection; it cannot change the
  model, which is checked against a kept count and a content hash.
- **The counts are capped at 4095** and halved on saturation. This bounds the
  reciprocal table and adds a slight recency weighting; a node visited millions of
  times is estimated from its recent 4095 visits.
- **The probability state is 11 bits.** Probabilities reach 2047/2048; the remaining
  floor is about 0.0007 bits per symbol, so a 2,000,000-byte run of 0x00 costs 410
  bytes.
- **Size cap.** Both directions share one limit: 2^30 symbols, i.e. 512 MiB of input.
  The encoder checks the file size before allocating and refuses anything larger.
- **Incompressible input expands, and how much depends on size.** There is no stored
  mode, so random bytes pay the model and the position channel with nothing to predict.
  The expansion is 13–14% at 10 KB, where the model costs about 0.117 bytes per input
  byte and only 10,000 positions are available to amortise it; 9.5% at 100 KB, the
  shallowest point; and 12% again at 1 MB, where the trie gains another context depth —
  the support threshold is reached one level deeper — and the kept set grows from 4,439
  nodes to 60,682, 13.7 times for ten times the input. The position channel itself is
  flat at about 1.018 bytes per input byte throughout, which is the floor the expansion
  cannot go below: with nothing to predict, the tail code carries every symbol.
- **A crafted size is bounded but not free.** A stream declaring 2^29 bytes, i.e.
  2^30 symbols, allocates 1 GB and takes on the order of fifteen seconds before it is
  rejected. The allocation is capped and wrapped.

## 7. Integrity

Two independent FNV-1a checksums, each verified before the region it protects is used:
the **header checksum** over the 32 header bytes, verified before any field is read,
and the **data checksum** over the payload, verified before any output is produced.

Beyond the checksums the decoder validates that the two section lengths equal the
payload size exactly, that the context length and the root fallback are in range, that
the decoded size is within a hard bound and can be allocated, and that the rebuilt
closure matches the encoder's kept count and an order-independent hash of every kept
node's `(depth, edge, answer)`. The section reader records whether it read past the end
of its stream, and the decoder rejects a stream whose declared size leaves the position
stream largely unread. A tail rank naming a symbol that is neither the answer, the
fallback nor one of the fourteen others is refused rather than emitted.

Because the two rates are learned rather than transmitted, they are not covered by any
checksum — they are not in the stream. Their correctness rests on the encoder and
decoder performing the same update in the same order, which the round-trip tests
exercise on every position of every file.

## 8. References

- E. Fredkin, "Trie memory", *Communications of the ACM* 3(9), 1960.
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, section 6.3: digital search
  trees and tries.
- D. R. Morrison, "PATRICIA — practical algorithm to retrieve information coded in
  alphanumeric", *Journal of the ACM* 15(4), 1968: path-compressed tries.
- P. M. McIlroy, K. Bostic, M. D. McIlroy, "Engineering radix sort", *Computing
  Systems* 6(1), 1993: the stable counting-sort passes used for the context keys.
- U. Manber, G. Myers, "Suffix arrays: a new method for on-line string searches",
  *SIAM Journal on Computing* 22(5), 1993: ordering positions by their context.
- J. Rissanen, "Modeling by shortest data description", *Automatica* 14(5), 1978: the
  MDL view behind support pruning, and behind the question of whether a statistic is
  worth transmitting or worth learning from traffic.
- R. E. Krichevsky, V. K. Trofimov, "The performance of universal encoding", *IEEE
  Transactions on Information Theory* 27(2), 1981: the `(hits + ½)/(spans + 1)`
  estimator and its `½·log₂(n)` redundancy.
- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., 2006,
  ch. 13: universal coding and the redundancy of sequential estimation.
- I. H. Witten, R. M. Neal, J. G. Cleary, "Arithmetic coding for data compression",
  *Communications of the ACM* 30(6), 1987: the range coder used for every channel.
- G. Fowler, L. C. Noll, K.-P. Vo, "Fowler/Noll/Vo hash", 1991: the FNV-1a checksums.
