# memorizer5

A memoriser compressor in which sixteen shared tail distributions replace the one
global tail model, so each node's residual statistics are transmitted as a 4-bit
pointer rather than as a distribution of its own.

```
memorizer5 c <input> <output>    compress
memorizer5 d <input> <output>    decompress
memorizer5 t <input> [S]         size, ratio, throughput, channels, round-trip check
```

C++11, ASCII-only source, in-memory I/O, no external libraries.

## 1. The method

Compression here means *transmit a model that reproduces the input, then repair the
positions the model gets wrong*. The model is a trie of contexts over a 4-bit symbol
stream:

- the input bytes are re-blocked into **nibbles**, so a 500 KB file becomes a million
  positions;
- every distinct context of length 1..Q (Q = 8) is a node;
- a node's **answer** is the symbol that most often follows its context.

Two reductions make that trie small enough to transmit. **Mode-preserving collapse**
drops a node whose answer equals its parent's, because a position whose deepest match is
dropped gets the same answer from its nearest surviving ancestor — no prediction
changes. **Support pruning** drops a node whose context occurs fewer than S = 15 times,
because a stated node costs model bytes while the misses it repairs are already cheap.

```
kept(n)    = n survives the collapse and support(n) >= S
closure(n) = kept(n), or n lies on a path from the root to a kept node
```

A closure node that is not kept is pure structure: it carries no answer, only an
"inherit" code, and exists so the decoder can reach a kept descendant.

## 2. Where the novelty is

A memoriser that states only an answer leaves the decoder unable to tell whether the
answer is right. The obvious repair is to transmit, alongside the model, a per-position
flag saying *right or wrong* and a residual saying *what it should have been*. Those two
channels then carry the entire cost of the model's self-assessment: one bit and one rank
for every position, almost all of it the model narrating its own mistakes one at a time.

Two better answers exist, and this design uses both.

**Learn the reliability instead of stating it.** Every kept node also states a
**fallback** — the symbol that most often replaces the answer when the answer is wrong —
and then a position asks three questions: *was it the answer*, *was it the fallback*,
*which of the others*. Neither rate needs to be transmitted: both sides count the
outcomes as they code, so the decoder reconstructs each node's confidence from the
positions it has already decoded.

**Share what a node learns about its tail.** The third question — the symbols that are
neither the answer nor the fallback — is not the same everywhere: a node's residual tail
has its own shape. But stating a tail distribution per node would cost more than it
saves.

The observation the design rests on: **the tails have a small number of shapes.** Cluster
the nodes by the shape of their residual tail and sixteen representative distributions
describe the corpus well. Each node then carries a **4-bit profile index** naming which
shape applies to it.

> **The novel step is this:** the residual tail is a per-node statistic that is
> transmitted as a *pointer into a shared table of sixteen shapes* rather than as a
> distribution of its own — and the table itself is never transmitted at all, because
> the sixteen shapes are adaptive trees in the coder that both sides grow from the
> positions they code.

The index is therefore the entire price. Sixteen shared shapes cost nothing to send; the
4-bit index costs 4 bits per kept node.

## 3. What it bought

Measured with the support threshold fixed at 8 on both sides, so that only the tail
coding differs:

| channel | one global tail tree | one tail tree per profile |
|---|---|---|
| C1 model | 58,275 B | 66,236 B |
| C3 positions | 728,555 B | **681,008 B** |
| total | 787,270 B | **747,684 B** |

The model grows by 7,961 coded bytes (the indices) and the position channel shrinks by
47,547. The delivered default is `S = 15`, a further 1,203 bytes better; the table above
fixes the threshold only to isolate the mechanism.

The sixteen shapes also let the threshold rise: sharing the tail made each repaired miss
cheaper, and a marginal node is worth less when the misses it prevents are worth less.

## 4. The algorithm

### 4.1 Construction by one radix sort

```
build(x[0..M-1], Q, S):
    # 1. pack the Q-symbol key of every position
    for t in Q..M-1:
        key[t] = pack(x[t-1], x[t-2], ..., x[t-Q])      # most recent in the high nibble

    # 2. one stable radix sort of the positions by key
    idx = radix_sort(positions Q..M-1 by key)

    # 3. every depth is already grouped: equal (d)-suffixes are a contiguous run of the
    #    (d+1)-ordered array, so shifting the key finds the boundaries
    for d in Q down to 1:
        for each maximal run of idx with equal (key >> 4*(Q-d)) & mask:
            groups[d].append(run_key, run_length, argmax_symbol(x[run]))

    # 4. kept(n) = mode differs from the parent's mode AND support >= S
    #    depth 1 has no parent group; its parent is the root, whose answer is rootMode
    for d in 1..Q:
        for each group g at depth d:
            parentMode = ( d == 1 ) ? rootMode
                                    : mode of binary_search(groups[d-1], g.key >> 4)
            kept[g] = (g.mode != parentMode) and (g.support >= S)

    # 5. closure = the trie built from the kept nodes; the builder creates the path
    #    nodes automatically, with no answer of their own
    closure = trie()
    for d in 1..Q:
        for each kept group g at depth d:
            node = descend(closure, g.key's d symbols)
            node.answer = g.mode
```

### 4.2 What each node learns

Two things are settled encoder-side, because both are properties of the whole input
rather than of a single position.

```
learn(x, closure, rootAnswer):
    # the fallback and the tail histogram, from one walk of the input
    for t in 1..M-1:
        n, answer = walk_to_node(closure, background of t, rootAnswer)
        if x[t] != answer:  miss[n][ rank_among_15(x[t], answer) ] += 1

    for each node n:
        # ties go to the smallest symbol, because the scan keeps the first maximum
        fb = smallest r with miss[n][r] maximal, or NONE if all are zero
        fbSym[n] = symbol of fb, or NONE            # undo the rank

        # the tail: the miss histogram with the fallback removed, re-ranked over the
        # symbols that remain. That is 14 when a fallback was removed and 15 when it
        # was not, so the profile space is 15 slots and a fallback node leaves slot 14
        # empty.
        tail[n] = miss[n] with fb removed, re-ranked

    # cluster the tail histograms into PROFILES shared shapes
    k-means over the nodes with a non-empty tail, cross-entropy objective,
    farthest-first initialisation from the heaviest node, twelve iterations,
    ties broken by the lowest index
    -> prof[n] in 0..15

    prof[root] = 0     # the root is not serialised with the closure, so both sides fix it
```

The clustering is a k-means whose distance is a coding cost: a profile is judged by how
cheaply it codes the tails given to it, which is the quantity that actually matters.
Farthest-first initialisation and the lowest-index tie-breaks make the result
deterministic, so the same input always yields the same profiles and the same file.

### 4.3 Coding a position

```
encode_positions(x, closure, rootAnswer):
    for every node: rate = {} , fbRate = {}          # both counts at zero
    for every profile: tailTree[p] = uniform          # adaptive, learned as we go

    for t in 1..M-1:
        n, answer = walk_to_node(...)
        fb        = fbSym[n]
        p         = prof[n]
        p0        = estimate(rate[n])
        p1        = estimate(fbRate[n])

        if x[t] == answer:      bit(p0, 0)
        else:
            bit(p0, 1)                      # a miss
            if fb != NONE:
                if x[t] == fb:  bit(p1, 0)
                else:           bit(p1, 1)
            if x[t] != fb or fb == NONE:
                tree(tailTree[p], 4, rank_of(x[t], excluding answer and fb))

        observe( rate[n],   x[t] == answer )
        if x[t] != answer and fb != NONE:
            observe( fbRate[n], x[t] == fb )

estimate(L)   = clamp( ((2*L.hit+1) * recip[L.span] + 2^20) >> 21 , 1, 2047 )
observe(L,ok) = L.hit += ok; L.span += 1;
                if L.span == COUNT_LIMIT:
                    L.hit = (L.hit + 1) >> 1        # ceil, so a hit never vanishes
                    L.span = (L.span + 1) >> 1
```

Decoding is the mirror image, and the two `observe` calls are the same lines in the same
order. That identity is the mechanism: if the encoder and decoder updated their counts
differently the two would diverge without any checksum noticing, because the counts are
not in the stream. The update is therefore deliberately the last thing in the loop body
on both sides. The same applies to the sixteen tail trees: they are adaptive, they are
not in the stream, and they must grow identically on both sides.

### 4.4 The estimator, without division

The estimate is the **Krichevsky–Trofimov** one, `(hits + ½)/(spans + 1)` — the natural
choice for a Bernoulli source with an unknown rate, with a total regret of
`½·log₂(n) + O(1)`. The coder wants an 11-bit probability, so the value needed at each
position is `2048·(2·hits+1)/(2·spans+2)`.

The denominator depends only on `spans`, and `spans` is capped, so the reciprocals are
tabulated once and multiplied in:

```
recip[s] = floor(2³² / (2s + 2))
p₁₁      = ((2·hits + 1) · recip[spans] + 2²⁰) >> 21
```

The shift of 21 is `32 − 11`. The truncated reciprocal makes the multiply agree with the
division exactly on 99.945% of the reachable `(hits, spans)` pairs and differ by one part
in 2048 on the rest — 0.0144 bits worst case, a rounding choice rather than an error, and
identical on both sides so it cannot affect a round trip. **No division appears on the
coding path.**

`COUNT_LIMIT = 4095` caps the counters. Once a node has been seen that many times its
estimate has stopped moving, so both counts are ceil-halved and counting continues: the
ratio survives, the magnitude does not.

### 4.5 The rank mapping

The excluded symbols are sorted ascending and both tests read the *original* symbol:

```
rank = symbol - (symbol > first) - (second exists and symbol > second)
```

and the inverse walks ascending, each test on the value produced so far:

```
symbol = rank; if (symbol >= first) symbol++; if (second exists and symbol >= second) symbol++
```

The alphabet is fourteen when a fallback is removed and fifteen when it is not, because a
node with no fallback excludes only its answer.

## 5. Vocabulary

- **symbol** — a 4-bit nibble; the byte stream is split into two per byte.
- **context** — the `d` symbols immediately before a position, most recent first.
- **answer / mode** — the most frequent successor of a context over the input.
- **collapse** — deleting a node whose answer equals its parent's answer.
- **kept node** — survives the collapse and meets the support threshold.
- **closure** — the kept nodes plus every node on a path to one.
- **support** — how many positions carry a given context.
- **fallback** — the symbol that most often replaced the answer among a node's misses.
  Transmitted once per node.
- **tail** — the symbols that are neither the answer nor the fallback.
- **profile** — one of sixteen shared tail shapes. A node carries a 4-bit index naming
  one; the shape itself is an adaptive tree the decoder grows as it decodes.
- **rate, fallback rate** — how often a node's answer has been right, and how often a
  miss has been its fallback. Learned, never transmitted.
- **estimate** — the Krichevsky–Trofimov value `(hits + ½)/(spans + 1)`, evaluated as a
  multiply by tabulated reciprocals.
- **C1, C3** — the two channels the output is accounted in: **C1** is the model (closure
  plus per-node statistics), **C3** is the position stream. Their sum plus the 32-byte
  header and two 4-byte checksums is the file size.

## 6. Measured results

Canterbury corpus, 2,810,784 bytes, each file compressed as a standalone input at the
delivered default `S = 15`. Throughput is in memory, excluding file I/O. Sizes are exact
and reproducible; the MB/s figures move a few percent with machine load.

| file | bytes | compressed | ratio | MB/s compress | MB/s decompress |
|---|---|---|---|---|---|
| alice29.txt | 152,089 | 55,267 | 0.3634 | 2.87 | 13.91 |
| asyoulik.txt | 125,179 | 48,577 | 0.3881 | 2.61 | 13.63 |
| cp.html | 24,603 | 11,015 | 0.4477 | 1.62 | 17.13 |
| fields.c | 11,150 | 5,086 | 0.4561 | 1.98 | 18.37 |
| grammar.lsp | 3,721 | 2,034 | 0.5466 | 1.46 | 19.08 |
| kennedy.xls | 1,029,744 | 236,751 | 0.2299 | 6.62 | 34.35 |
| lcet10.txt | 426,754 | 136,373 | 0.3196 | 3.12 | 11.67 |
| plrabn12.txt | 481,861 | 172,053 | 0.3571 | 3.05 | 11.32 |
| ptt5 | 513,216 | 57,922 | 0.1129 | 7.88 | 36.13 |
| sum | 38,240 | 18,760 | 0.4906 | 1.62 | 18.06 |
| xargs.1 | 4,227 | 2,643 | 0.6253 | 1.24 | 16.71 |
| **total** | **2,810,784** | **746,481** | **0.2656** | **4.25** | **19.02** |

The support threshold, measured over the whole corpus on this design:

| S | 4 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | **15** | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total bytes | 765,185 | 753,488 | 747,684 | 746,991 | 747,033 | 748,250 | 748,377 | 748,538 | 747,847 | **746,481** | 747,686 |

The surface is flat and not monotone: the whole range from 8 to 16 spans 2,057 bytes,
0.28% of the output, and the ordering within it is not a smooth function of S. The
delivered value is the measured minimum, but the message is that the threshold sits on a
broad plateau — it is not a finely tuned parameter, and no particular optimum should be
read into it.

All eleven files round-trip byte-exactly. Edge cases verified: empty, 1, 2 and 3 bytes, a
2,000,000-byte run of 0x00 (410 B), a 1,000,000-byte run of 0xFF (230 B), an odd-length
100,001-byte file, and 100,000 random bytes. Two compressions of the same input are
byte-identical.

| channel | bytes | share |
|---|---|---|
| C1 model | 43,284 | 5.8% |
| C3 positions | 702,757 | 94.1% |
| header and two checksums | 440 | 0.06% |

The model is 5.8% of the output: 2.40 bytes per kept node across the corpus's 18,021 kept
nodes. It states the closure's shape, its answers, its fallbacks and its profile indices —
and nothing about its reliability, which the decoder counts for itself.

## 7. Known limits

- **The fallback symbol is transmitted.** It must be known before the position is coded,
  so it cannot be learned from that same position. It could be learned across positions,
  at the cost of coding the early ones against a symbol that is not yet right.
- **The profiles are sixteen, not per-node.** A node whose tail is unlike any of the
  sixteen shared shapes is coded worse than its own distribution would allow. More shapes
  would recover some of that, at a wider index on every node.
- **The clustering costs compression time.** The k-means runs once per file over the nodes
  that have a tail, and it is why compression is slower than decompression by a factor of
  four. It is encoder work only; the decoder just reads the index.
- **Incompressible input expands, by an amount that depends on size and on the seed.**
  There is no stored mode. Measured over several seeds: about 7% at 10 KB, 9% at 100 KB,
  13% at 1 MB, and 8% at 2 MB. The variation with size comes from the model
  being amortised over more positions while the trie gains context depth; the variation
  between seeds at a fixed size is several percent.
- **The counts are capped at 4095** and ceil-halved on saturation, which bounds the
  reciprocal table and adds a slight recency weighting.
- **The probability state is 11 bits.** Probabilities reach 2047/2048; the remaining floor
  is about 0.0007 bits per symbol, so a 2,000,000-byte run of 0x00 costs 410 bytes.
- **Size cap.** Both directions share one limit: 2^30 symbols, i.e. 512 MiB of input. The
  encoder checks the file size before allocating and refuses anything larger. The encoder
  needs about 24 bytes per input byte for the radix sort, so the cap implies roughly
  12 GiB of working set; an allocation failure is caught and reported.
- **`c` and `d` do not take a threshold argument.** Only `t <input> [S]` accepts one, so
  the threshold used to produce a file is always the compiled default. `t` reports the
  better of two timed repetitions, which is why its MB/s figures are slightly optimistic.

## 8. Integrity

Two independent FNV-1a checksums, each verified before the region it protects is used: the
**header checksum** over the 32 header bytes (the magic is compared first, then the
checksum over all 32 before any other field is used), and the **data checksum** over the
payload, verified before any output is produced.

Beyond the checksums the decoder validates that the two section lengths equal the payload
size exactly, that the context length and the root fallback are in range, that the decoded
size is within a hard bound and can be allocated, and that the rebuilt closure matches the
encoder's kept count and an order-independent hash of every kept node's
`(depth, edge, answer)`. The section reader records whether it read past the end of its
stream, and the decode loop checks that flag as it goes rather than only at the end. A
tail rank naming a symbol that is neither the answer, the fallback nor one of the remaining
symbols is refused rather than emitted.

The declared position count is also bounded by what the stream could possibly code: the
most confident bit the coder can represent costs 0.000705 bits, so no stream of `l3` bytes
can code more than about 11,300 positions per byte, and a declared size beyond that is
rejected before any buffer is allocated. Together with the in-loop exhaustion check, a
short forged stream cannot make the decoder do a large amount of work or reserve a large
buffer. It does not remove the amplification: a crafted container of about 90 KB can still
ask for the full 1 GiB buffer and be rejected after several seconds. What the bounds do is
cap the amplification at roughly 12,000 times the container rather than leaving it
unbounded.

**What the checksums do not do.** FNV-1a is unkeyed: it detects corruption, not tampering.
Anyone who rewrites a header field and recomputes the checksum gets a stream the decoder
accepts, and two header fields are not independently evidenced:

- **`firstNib`** is the first symbol of the decoded stream, so changing it produces a file
  whose first nibble differs and which is otherwise valid — though some values are
  rejected, because the resulting stream must still decode;
- **`srcLen`** is the declared position count. Within the slack the coder allows, changing
  it changes how many positions are decoded from the same bytes, and a highly predictable
  stream can absorb many extra positions without exhausting. The tolerated difference is
  data-dependent and can be large: it is a few positions on ordinary text and thousands on
  a highly predictable stream.

Rejecting such a stream would require redundancy rather than a checksum — a second copy of
the size, or a keyed digest. This is an accidental-corruption design: the round-trip tests
exercise every position of every file, but a deliberately re-checksummed file is a
different valid stream rather than a rejected one.

Because the rates and the sixteen profile trees are learned rather than transmitted, they
are not covered by any checksum — they are not in the stream. Their correctness rests on
the encoder and decoder performing the same updates in the same order.

## 9. References

- E. Fredkin, "Trie memory", *Communications of the ACM* 3(9), 1960.
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, section 6.3: digital search
  trees and tries.
- D. R. Morrison, "PATRICIA — practical algorithm to retrieve information coded in
  alphanumeric", *Journal of the ACM* 15(4), 1968: path-compressed tries.
- P. M. McIlroy, K. Bostic, M. D. McIlroy, "Engineering radix sort", *Computing Systems*
  6(1), 1993: the stable counting-sort passes used for the context keys.
- U. Manber, G. Myers, "Suffix arrays: a new method for on-line string searches", *SIAM
  Journal on Computing* 22(5), 1993: ordering positions by their context.
- J. Rissanen, "Modeling by shortest data description", *Automatica* 14(5), 1978: the MDL
  view behind support pruning, and behind the question of whether a statistic is worth
  transmitting or worth learning from traffic.
- R. E. Krichevsky, V. K. Trofimov, "The performance of universal encoding", *IEEE
  Transactions on Information Theory* 27(2), 1981: the `(hits + ½)/(spans + 1)` estimator
  and its `½·log₂(n)` redundancy.
- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., 2006, ch. 13:
  universal coding and the redundancy of sequential estimation.
- S. P. Lloyd, "Least squares quantization in PCM", *IEEE Transactions on Information
  Theory* 28(2), 1982: the clustering, and its convergence.
- Y. Linde, A. Buzo, R. M. Gray, "An algorithm for vector quantizer design", *IEEE
  Transactions on Communications* 28(1), 1980: the k-means construction used to derive
  the shared tail profiles.
- P. F. Brown, P. V. deSouza, R. L. Mercer, V. J. Della Pietra, J. C. Lai, "Class-based
  n-gram models of natural language", *Computational Linguistics* 18(4), 1992: sharing
  statistics across contexts that behave alike, the idea the profiles exploit.
- I. H. Witten, R. M. Neal, J. G. Cleary, "Arithmetic coding for data compression",
  *Communications of the ACM* 30(6), 1987: the range coder used for every channel.
- G. Fowler, L. C. Noll, K.-P. Vo, "Fowler/Noll/Vo hash", 1991: the FNV-1a checksums.
