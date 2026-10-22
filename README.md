# memorizer6

A memoriser compressor that ships a small model, learns how far to trust it, and keeps
that trust on a fixed memory of recent outcomes rather than on a lifetime average.

```
memorizer6 c <input> <output>    compress
memorizer6 d <input> <output>    decompress
memorizer6 t <input> [S]         size, ratio, throughput, channels, round-trip check
```

C++11, ASCII-only source, in-memory I/O, no external libraries. Build:

```
g++ -O3 -s -std=c++11 -pedantic-errors -Wall -Wextra -o memorizer6.exe memorizer6.cpp
```

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
dropped gets the same answer from its nearest surviving ancestor, and no prediction
changes. **Support pruning** drops a node whose context occurs fewer than S = 15 times,
because a stated node costs model bytes while the misses it repairs are already cheap.

```
kept(n)    = n survives the collapse and support(n) >= S
closure(n) = kept(n), or n lies on a path from the root to a kept node
```

A closure node that is not kept is pure structure: it carries no answer, only an
"inherit" code, and exists so the decoder can reach a kept descendant.

The threshold is worth being precise about, because it is not a crude stand-in for a
sharper test that this design failed to implement. The sharper test would keep a node
when the bits its own positions save exceed the bytes it costs, and that test is easy to
state and easy to compute from the node's own histogram. Measured, it is much worse: over
a range of charges per node it produced 1.19 to 1.25 MB on this corpus, against 734 KB
here. The reason is structural. A node's answer is not spent only on its own positions.
It is also the **fallback** for every position beneath it, and that fallback is what makes
the third question cheap, so a node's value grows with the number of positions that pass
through it. Support is exactly that number. A count threshold measures subtree exposure;
a profit test on a node's own positions measures something smaller and gets the sign of
marginal nodes wrong.

## 2. Where the novelty is

A memoriser that states only an answer leaves the decoder unable to tell whether the
answer is right. The obvious repair is to transmit, alongside the model, a per-position
flag saying *right or wrong* and a residual saying *what it should have been*. Those two
channels then carry the entire cost of the model's self-assessment: one bit and one rank
for every position, almost all of it the model narrating its own mistakes one at a time.

Two better answers exist, and this design uses both.

**Learn the reliability instead of stating it.** Every kept node also states a
**fallback** (the symbol that most often replaces the answer when the answer is wrong),
and then a position asks three questions: *was it the answer*, *was it the fallback*,
*which of the others*. Neither rate needs to be transmitted: both sides count the
outcomes as they code, so the decoder reconstructs each node's confidence from the
positions it has already decoded.

**Share what a node learns about its tail.** The third question (the symbols that are
neither the answer nor the fallback) is not the same everywhere: a node's residual tail
has its own shape. But stating a tail distribution per node would cost more than it
saves.

The observation the design rests on: **the tails have a small number of shapes.** Cluster
the nodes by the shape of their residual tail and sixteen representative distributions
describe the corpus well. Each node then carries a **4-bit profile index** naming which
shape applies to it.

> **The first novel step is this:** the residual tail is a per-node statistic that is
> transmitted as a *pointer into a shared table of sixteen shapes* rather than as a
> distribution of its own, and the table itself is never transmitted at all, because the
> sixteen shapes are adaptive trees in the coder that both sides grow from the positions
> they code.

The index is therefore the entire price. Sixteen shared shapes cost nothing to send; the
4-bit index costs 4 bits per kept node.

> **The second novel step is this:** the rate that says how far to trust a node is a
> *fixed-memory* statistic, and the length of that memory is a first-class design
> parameter, not a rounding limit. A counted average that never forgets cannot follow
> local predictability; a short memory cannot express near-certainty. One rule gives
> both: the counts are halved when they fill, except while the node has never yet been
> wrong, which is the only case where halving would destroy the evidence of certainty.

## 3. The shared tail profiles, measured against one global tail

Both columns are this code at the delivered support threshold, with the profile count
changed from sixteen to one so that the shapes collapse into a single shared tail tree:

| channel | one global tail tree | sixteen shared shapes |
|---|---|---|
| C1 model | 38,259 B | 43,284 B |
| C3 positions | 740,657 B | **690,292 B** |
| total | 779,356 B | **734,016 B** |

The model grows by 5,025 bytes, which is the sixteen indices, and the position channel
shrinks by 50,365. Sharing the tail also makes each repaired miss cheaper, which is what
lets the support threshold sit as high as it does: a marginal node is worth less when the
misses it prevents are worth less.

## 4. The estimator's memory

The two learned rates are the whole of the model's self-assessment, so their memory
length is not a detail. A rate built by counting every outcome since the file began has
three faults at once: it treats a context that was right nine times in ten a million
positions ago as nine-in-ten reliable now; it stops moving once the counts are large; and
its counts grow without bound, which forces either a wide counter or a table that no
longer fits in cache.

The repair is to cap the counts and halve both when the cap is reached. The ratio
survives, the magnitude does not, and the estimate decays geometrically: with the cap at
80, evidence older than about 40 outcomes counts for less than half. Swept over the whole
corpus, one row per rebuild of this source with the constant changed:

| COUNT_LIMIT | 32 | 40 | 48 | 56 | 64 | 72 | **80** | 88 |
|---|---|---|---|---|---|---|---|---|
| total bytes | 738,510 | 735,376 | 734,602 | 734,407 | 734,240 | 734,115 | **734,016** | 734,201 |

The surface is flat between 48 and 88 (586 bytes across that whole range), and the
delivered value is the measured minimum. Compared with a cap in the thousands, which is
where a counted estimator naturally sits, the short memory is worth 1.7% of the output:
most of the predictability in real inputs is local, and a long memory averages it away.

There is one case where a short memory is wrong, and it cannot be fixed by choosing a
better cap. Halving a counter that has been right every single time caps the estimate at
41/42 = 0.988, worth 0.017 bits per position, where an unhalved counter reaches 2047/2048
= 0.9995, worth 0.0007. A run of identical bytes would pay the difference on every
position: with a single 80-cap, a two-megabyte run of 0x00 costs 6,578 bytes instead of
428. A perfect record is the one form of evidence that halving destroys rather than
discounts, so a node that has never yet been wrong keeps counting to 4095; its first miss
drops it to 80 and it stays there, because from that point the useful question is what the
node has been doing lately. Both tests sit inside the branch that runs when a counter
fills, so the path every position takes is one comparison and one increment.

### 4.1 Other conditionings, and what they cost

Three other ways of conditioning the answer estimate were measured against this one on
five of the corpus files, scoring only the bits of the answer decision over identical
positions and identical models.

**Splitting a node's counts by the outcome of its previous visit** costs between 12% and
76% more. The split is not useless, but each of the two cells then sees half the evidence,
and the estimator's redundancy grows faster than the correlation repays.

**Splitting the estimate by depth and by the predicted symbol**, as a shared prior for
young nodes, moves the cost by under one percent in either direction. There is almost
nothing for it to fix: only 0.4% to 12.5% of positions land on a node with fewer than
eight prior visits.

**An exponential rate in the coder's own probability space**, forgetting by a factor of
two every k observations, is a close relative of what the short cap does. Measured on the
same five files it beat the counted estimate on the two most predictable of them, by up to
13% of the answer channel, and lost on the three text files by 1% to 4%. The difference
that matters is that a count knows how much evidence it has and a decaying probability
does not, which is why a count is what the estimator is built on. The two places the
exponential wins are the two the counted estimator has to be fixed to compete, and both
are described above: local adaptation comes from the short cap, and near-certainty from
letting a never-wrong counter keep counting.

None of the three is in the design.

## 5. The algorithm

### 5.1 Construction by one radix sort

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

### 5.2 What each node learns

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

### 5.3 Coding a position

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
observe(L,ok) = L.hit += ok; L.span += 1
                if L.span >= COUNT_LIMIT and ( L.hit < L.span or L.span >= HARD_LIMIT ):
                    L.hit  = (L.hit + 1) >> 1      # ceil, so a hit never vanishes
                    L.span = (L.span + 1) >> 1
```

Decoding is the mirror image, and the two `observe` calls are the same lines in the same
order. That identity is the mechanism: if the encoder and decoder updated their counts
differently the two would diverge without any checksum noticing, because the counts are
not in the stream. The update is therefore deliberately the last thing in the loop body
on both sides. The same applies to the sixteen tail trees: they are adaptive, they are
not in the stream, and they must grow identically on both sides.

### 5.4 The estimator, without division

The estimate is the **Krichevsky-Trofimov** one, `(hits + 1/2)/(spans + 1)`, the natural
choice for a Bernoulli source with an unknown rate, with a total regret of
`(1/2)*log2(n) + O(1)`. The coder wants an 11-bit probability, so the value needed at each
position is `2048*(2*hits+1)/(2*spans+2)`.

The denominator depends only on `spans`, and `spans` is capped, so the reciprocals are
tabulated once and multiplied in:

```
recip[s] = floor(2^32 / (2s + 2))
p11      = ((2*hits + 1) * recip[spans] + 2^20) >> 21
```

The shift of 21 is `32 - 11`. The truncated reciprocal makes the multiply agree with the
division exactly on 99.945% of the reachable `(hits, spans)` pairs and differ by one part
in 2048 on the rest, which is 0.0144 bits worst case: a rounding choice rather than an
error, and identical on both sides so it cannot affect a round trip. **No division appears
on the coding path.**

`COUNT_LIMIT = 80` bounds the memory of an imperfect counter and `HARD_LIMIT = 4095` that
of a counter that has never been wrong, as described in section 4. The reciprocal table
spans the hard limit, so it holds 4,096 entries and is built once, read-only afterwards.

### 5.5 The rank mapping

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

## 6. Vocabulary

- **symbol**: a 4-bit nibble; the byte stream is split into two per byte.
- **context**: the `d` symbols immediately before a position, most recent first.
- **answer / mode**: the most frequent successor of a context over the input.
- **collapse**: deleting a node whose answer equals its parent's answer.
- **kept node**: survives the collapse and meets the support threshold.
- **closure**: the kept nodes plus every node on a path to one.
- **support**: how many positions carry a given context.
- **fallback**: the symbol that most often replaced the answer among a node's misses.
  Transmitted once per node.
- **tail**: the symbols that are neither the answer nor the fallback.
- **profile**: one of sixteen shared tail shapes. A node carries a 4-bit index naming one;
  the shape itself is an adaptive tree the decoder grows as it decodes.
- **rate, fallback rate**: how often a node's answer has been right lately, and how often
  a miss has been its fallback lately. Learned, never transmitted.
- **memory**: how many recent outcomes a rate is built from. Set by `COUNT_LIMIT` once a
  node has been wrong at least once, and by `HARD_LIMIT` while it never has.
- **estimate**: the Krichevsky-Trofimov value `(hits + 1/2)/(spans + 1)`, evaluated as a
  multiply by tabulated reciprocals.
- **C1, C3**: the two channels the output is accounted in. **C1** is the model (closure
  plus per-node statistics), **C3** is the position stream. Their sum plus the 32-byte
  header and two 4-byte checksums is the file size.

## 7. Measured results

Canterbury corpus, 2,810,784 bytes, each file compressed as a standalone input at the
delivered default `S = 15`. Sizes are exact and reproducible; the MB/s figures move a few
percent with machine load.

| file | bytes | compressed | ratio |
|---|---|---|---|
| alice29.txt | 152,089 | 55,245 | 0.3632 |
| asyoulik.txt | 125,179 | 48,589 | 0.3882 |
| cp.html | 24,603 | 11,009 | 0.4475 |
| fields.c | 11,150 | 5,032 | 0.4513 |
| grammar.lsp | 3,721 | 2,022 | 0.5434 |
| kennedy.xls | 1,029,744 | 227,071 | 0.2205 |
| lcet10.txt | 426,754 | 136,040 | 0.3188 |
| plrabn12.txt | 481,861 | 172,292 | 0.3576 |
| ptt5 | 513,216 | 55,633 | 0.1084 |
| sum | 38,240 | 18,446 | 0.4824 |
| xargs.1 | 4,227 | 2,637 | 0.6238 |
| **total** | **2,810,784** | **734,016** | **0.2611** |

Throughput on the same corpus, in memory and excluding file I/O: about 3.1 MB/s compress
and 18.9 MB/s decompress. Almost all of the gap is the tail clustering, which runs once
per file on the encoder side only; the decoder reads the finished index.

The support threshold, measured over the whole corpus at the delivered memory:

| S | 4 | 6 | 8 | 10 | 12 | **15** | 18 | 22 |
|---|---|---|---|---|---|---|---|---|
| total bytes | 753,842 | 741,824 | 735,800 | 734,961 | 736,176 | **734,016** | 736,516 | 740,945 |

The ordering is not a smooth function of S: the whole range from 8 to 22 spans 6,929 bytes
and the points in between interleave. The delivered value is the measured minimum, but the
message is that the threshold sits on a broad plateau and no particular optimum should be
read into it.

All eleven files round-trip byte-exactly. Edge cases verified: empty, 1, 2 and 3 bytes
(50, 50, 50 and 52 bytes), a 2,097,152-byte run of 0x00 (428 B), a 1,048,576-byte run of
0xFF (239 B), an odd-length 100,001-byte file (1,790 B), and 100,000 pseudo-random bytes.
Two compressions of the same input are byte-identical.

| channel | bytes | share |
|---|---|---|
| C1 model | 43,284 | 5.9% |
| C3 positions | 690,292 | 94.1% |
| header and two checksums | 440 | 0.06% |

The model is 5.9% of the output: 2.40 bytes per kept node across the corpus's 18,021 kept
nodes. It states the closure's shape, its answers, its fallbacks and its profile indices,
and nothing about its reliability, which the decoder counts for itself.

## 8. Known limits

- **The fallback symbol is transmitted.** It must be known before the position is coded,
  so it cannot be learned from that same position. It could be learned across positions,
  at the cost of coding the early ones against a symbol that is not yet right.
- **The profiles are sixteen, not per-node.** A node whose tail is unlike any of the
  sixteen shared shapes is coded worse than its own distribution would allow. More shapes
  would recover some of that, at a wider index on every node.
- **The clustering costs compression time.** The k-means runs once per file over the nodes
  that have a tail, and it is why compression is slower than decompression by a factor of
  six. It is encoder work only.
- **A short memory costs a little speed.** The counts fill and halve once per forty
  outcomes per counter instead of once per few thousand. A build of this code with the cap
  raised to 4095 decompresses three to four percent faster; compression is unchanged
  within measurement noise. That is the price of the 1.7% it buys in size, on a decoder
  that runs six times faster than the encoder.
- **Incompressible input expands.** There is no stored mode. Measured with one fixed seed:
  6.9% at 10 KB, 8.8% at 100 KB and 12.5% at 1 MB. Most of the growth with size is the
  model being amortised over more positions while the trie gains depth.
- **The certainty path needs a long counter.** An estimate can only reach 2047/2048 while
  its counter has never been wrong, because any miss drops that counter to the 80-outcome
  memory. A stream that is almost but not exactly predictable therefore codes at the
  short-memory confidence, about 0.01 bits per position at best.
- **The probability state is 11 bits.** Probabilities reach 2047/2048; the remaining floor
  is 0.0007 bits per symbol, which is why a 2,097,152-byte run of 0x00 costs 428 bytes
  rather than fewer.
- **Size cap.** Both directions share one limit: 2^30 symbols, i.e. 512 MiB of input. The
  encoder checks the file size before allocating and refuses anything larger. The encoder
  needs about 24 bytes per input byte for the radix sort, so the cap implies roughly
  12 GiB of working set; an allocation failure is caught and reported.
- **`c` and `d` do not take a threshold argument.** Only `t <input> [S]` accepts one, so
  the threshold used to produce a file is always the compiled default. `t` reports the
  better of two timed repetitions, which is why its MB/s figures are slightly optimistic.

## 9. Integrity

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
  whose first nibble differs and which is otherwise valid, though some values are rejected
  because the resulting stream must still decode;
- **`srcLen`** is the declared position count. Within the slack the coder allows, changing
  it changes how many positions are decoded from the same bytes, and a highly predictable
  stream can absorb many extra positions without exhausting. The tolerated difference is
  data-dependent and can be large: it is a few positions on ordinary text and thousands on
  a highly predictable stream.

Rejecting such a stream would require redundancy rather than a checksum, such as a second
copy of the size or a keyed digest. This is an accidental-corruption design: the
round-trip tests exercise every position of every file, but a deliberately re-checksummed
file is a different valid stream rather than a rejected one.

Because the rates and the sixteen profile trees are learned rather than transmitted, they
are not covered by any checksum, since they are not in the stream. Their correctness rests
on the encoder and decoder performing the same updates in the same order.

## 10. References

- E. Fredkin, "Trie memory", *Communications of the ACM* 3(9), 1960.
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, section 6.3: digital search trees
  and tries.
- D. R. Morrison, "PATRICIA: practical algorithm to retrieve information coded in
  alphanumeric", *Journal of the ACM* 15(4), 1968: path-compressed tries.
- P. M. McIlroy, K. Bostic, M. D. McIlroy, "Engineering radix sort", *Computing Systems*
  6(1), 1993: the stable counting-sort passes used for the context keys.
- U. Manber, G. Myers, "Suffix arrays: a new method for on-line string searches", *SIAM
  Journal on Computing* 22(5), 1993: ordering positions by their context.
- J. Rissanen, "Modeling by shortest data description", *Automatica* 14(5), 1978: the MDL
  view behind support pruning, and behind the question of whether a statistic is worth
  transmitting or worth learning from traffic.
- R. E. Krichevsky, V. K. Trofimov, "The performance of universal encoding", *IEEE
  Transactions on Information Theory* 27(2), 1981: the `(hits + 1/2)/(spans + 1)`
  estimator and its `(1/2)*log2(n)` redundancy.
- L. D. Davisson, "Universal noiseless coding", *IEEE Transactions on Information Theory*
  19(6), 1973: estimating a source's statistics online, which is what the learned rates do.
- A. Moffat, "Implementing the PPM data compression scheme", *IEEE Transactions on
  Communications* 38(11), 1990: rescaling counts when they reach a maximum, the mechanism
  that gives the counts here their finite memory, and the memory-length trade behind it.
- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., 2006, ch. 13:
  universal coding and the redundancy of sequential estimation.
- S. P. Lloyd, "Least squares quantization in PCM", *IEEE Transactions on Information
  Theory* 28(2), 1982: the clustering, and its convergence.
- Y. Linde, A. Buzo, R. M. Gray, "An algorithm for vector quantizer design", *IEEE
  Transactions on Communications* 28(1), 1980: the k-means construction used to derive the
  shared tail profiles.
- P. F. Brown, P. V. deSouza, R. L. Mercer, V. J. Della Pietra, J. C. Lai, "Class-based
  n-gram models of natural language", *Computational Linguistics* 18(4), 1992: sharing
  statistics across contexts that behave alike, the idea the profiles exploit.
- I. H. Witten, R. M. Neal, J. G. Cleary, "Arithmetic coding for data compression",
  *Communications of the ACM* 30(6), 1987: the range coder used for every channel.
- G. Fowler, L. C. Noll, K.-P. Vo, "Fowler/Noll/Vo hash", 1991: the FNV-1a checksums.
