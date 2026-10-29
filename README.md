# Introduction

This repository presents a novel information-trie structure that can
efficiently memorize data and subsequently recall it, provided that the model
is accompanied by a "residual" data channel: the elements of the data that the
model fails to predict. Since this trie naturally generalizes the data used to
construct it, its held-out prediction rate is substantial, especially for 2-
and 1-bit symbol streams.

`memorizer_base.cpp` provides the baseline implementation; it is the best way
to study and understand how it works. To avoid conflating it with other
"memorizer" tries, the proposed name for this trie structure is
the Thales trie (or by author's name if that's culturally adequate).

The model can be created in independent threads and then radix-merged.
In addition, true multithreaded decompression is naturally possible if the
output stores the initial context (just 4 bytes) for each thread. Experiments
show that running node-level symbol statistics (which constitute
self-contained context modeling) permit recovering up to 40% of misses at
a small model-size cost. Fast integer leaky integrators are adequate for local
contexts; in fact, they are already used to model the probabilities of residue
symbols for the range encoder in `memorizer9.cpp`, and they can be
SIMD-optimized.

Additionally, the hit/miss bit stream and residue symbols can be combined
(to form full W-bit symbols) and then passed to the LZ77 match finder, which
can use the built trie nodes to store offsets (instead of using a hash table):
tests demonstrate 40-70% coverage of such full-W residues by LZ matches.
If encoding time constraints permit, the entropy of the hit/miss bit stream
can be reduced by 10–20% with a secondary 1-bit memorizer with Q = 6 to 10
(its output simply needs to be XORed with the original bit stream; no residue
of its own needs to be stored).

`memorizer9.cpp` is a practical implementation of a single-file compressor
based on the memorizer trie model. It achieves PPM compression levels at
comparable and sometimes higher speeds.

## Inception of Memorizer

The author's (Aleksey Vaneev) precursor idea *and* preliminary implementation
addressed the following problem: how to map a given input and its contexts to
a constellation of N-dimensional points so that a specific traversal of these
points, without knowing the context, reconstructs or "recalls" the context,
with each point mapping to a symbol. The initial solution was based on
floating-point vectors, and it turned out to memorize 99.5% of the context and
to require fewer points than there were input symbols, so it generalized to
some degree.

The next step was to discretize the vector into -1 and 1 values - binary
vectors. This yielded lower memorization precision but permitted the model to
be mapped to fixed-width integers. As you can imagine, a set of N-dimensional
points with only a symbol constraint can be represented by an infinite number
of sets of unique coordinate vectors. Thus, one can map a binary vector to an
exact input context as uniquely as by using any other uniqueness rule. So, in
its final form, the memorizer forms a constellation of context-points with
binary coordinates, each mapped to a specific symbol. It is unbelievably
simple in the end. The complexity lies in making it work fast.

The inevitable outcome of the inexactness of reconstruction is the residual
channel; without filling the gaps, the full output cannot be reconstructed.
But the means of carrying this residual channel are not specific - even ECC
can be used.

And to set the record straight, this is not context modeling (AI systems often
conflate the concepts). The Memorizer method does not require probability
evaluations at all (the "first in" symbol can be used as the node answer).
The most effective approach to assigning a symbol to a context-point (node) is
the weighted "majority vote" rule: the symbol that follows a given context
most often should be assigned to the context-point (weighted Boyer-Moore
balance).

## memorizer9

A memoriser compressor that transmits a trie of contexts and nothing else: the model
channel carries the shape of the trie, and the input is coded as one hit/miss bit per
position plus one residue symbol on a miss. Every probability the coder uses is an integer
recursion over symbols both sides have already seen, so no rate, table or count is ever
written to the stream.

```
memorizer9 c <input> <output>    compress
memorizer9 d <input> <output>    decompress
memorizer9 t <input> [S]         size, ratio, throughput, channels, round-trip check
```

C++11, ASCII-only source, in-memory I/O, no external libraries. Build:

```
g++ -O3 -s -std=c++11 -pedantic-errors -Wall -Wextra -o memorizer9.exe memorizer9.cpp
```

## 1. The method

Compression here means *transmit a model that reproduces the input, then say where the
model was wrong*. The model is a trie of contexts over a 4-bit symbol stream:

- the input bytes are re-blocked into **nibbles**, so a 500 KB file becomes a million
  positions;
- every distinct context of length 1..Q (Q = 8) is a node;
- a node's **answer** is the symbol that most often follows its context, ties to the
  smallest symbol.

Two reductions shrink that trie before it is sent. **Mode-preserving collapse** drops a
node whose answer equals its parent's, which removes a node without removing a prediction
from the build's view of the input. **Support pruning** drops a context seen fewer than S
times, because a stated node costs model bytes while the misses it repairs are already
cheap. Both are size reductions rather than semantic ones: at coding time the deepest
match gets a predictor of its own, learned from traffic, so a dropped node sends its
positions to an ancestor whose predictor differs, and the totals in section 6 measure that cost
instead of assuming it away.

```
kept(n)    = n survives the collapse and support(n) >= S
closure(n) = kept(n), or n lies on a path from the root to a kept node
```

The closure is the trie that is actually transmitted. A node that is not kept still exists
in it, still gets traversed, and still predicts: it is there so the decoder can reach a
kept descendant, and it is the predictor for every position that stops at it.

## 2. Where the novelty is

A memoriser that states an answer leaves the decoder unable to say whether the answer is
right, so the obvious repair is to ship a per-position flag saying *right or wrong* and a
residual saying *what it should have been*. Those two channels then carry the whole cost
of the model's self-assessment, one flag bit per coded position and one rank per miss.

The observation this design rests on: **the answer does not have to be transmitted, and
neither do the probabilities that make it credible.** Both sides see the same symbols in
the same order, so both can keep the same picture of a context and reach the same
prediction without a byte being spent on it.

> **The novel step is this:** the model channel carries only the *shape* of the trie,
> while the prediction and the confidence in it are produced by three integer recursions
> over already-decoded symbols. A node's mode is the heaviest of sixteen successor weights
> decayed by 1/32 per visit, so the predictor tracks what the context has been doing
> lately rather than what it did on average. The probability that prices the hit/miss bit
> is a leaky integrator, moved toward the outcome by a fixed fraction of the remaining
> distance, so it needs no counter, no ratio, no division and no table. Nothing about
> reliability, or even about which symbol is predicted, is in the stream.

The consequence is a model channel of 0.71 bytes per node and a position channel of two
decisions: was the mode right, and if not, which of the other fifteen symbols was it.

## 3. What it costs

Measured on the corpus, per position and per node:

| | |
|---|---|
| position channel | 5,403,296 bits over 5,621,557 positions = **0.9612 bits/position** |
| model channel | 126,840 bits over 22,331 nodes = **5.68 bits/node** |
| hits | 81.7% of positions are predicted correctly |

Two questions, two prices. The hit/miss bit is the expensive one, because a bit train that
is 82% zeros, because a hit is coded as 0, is not nearly free: at one probability for the whole file it would cost 0.68
bits per bit, and only per-context estimates bring it below 0.5. The residue is the other
side of the same coin: roughly one position in five reaches it, and each costs a few bits
through the node's own adaptive tree.

The residue is deliberately coded by a plain sixteen-slot adaptive tree per node, with no
model of its own beyond its own decays. That is the simplest thing that is correct and
symmetric, and it is where the remaining headroom sits: sharing statistics across nodes,
or weighting the tree by recency, would be a modelling addition rather than a structural
one.

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

    # 4. kept(n) = the mode differs from the parent's mode AND support >= S
    #    depth 1 has no parent group; its parent is the root, whose mode is global
    for d in 1..Q:
        for each group g at depth d:
            parentMode = ( d == 1 ) ? rootMode
                                    : mode of binary_search(groups[d-1], g.key >> 4)
            kept[g] = (g.mode != parentMode) and (g.support >= S)

    # 5. closure = the trie of the kept nodes; the builder creates path nodes with no
    #    answer of their own, so the decoder can reach a kept descendant
    closure = trie()
    for d in 1..Q:
        for each kept group g at depth d:
            node = descend(closure, g.key's d symbols)
```

The modes are encoder-side only: they decide which nodes survive, and they are never
written.

### 4.2 Coding a position

```
encode(x, closure):
    for every node: w[16] = 0, rate = 1/2, res[16] = 1024 each

    for t in 1..M-1:
        n     = deepest node whose edge path matches the context of t
        m     = n.mode()                     # heaviest weight, ties to the lowest symbol

        if m < 0:                            # nothing is known here yet
            tree(n.res, 4, x[t])             # the symbol, with no hit/miss bit at all
        else:
            hit = ( x[t] == m )
            bit( n.rate.prob(), hit ? 0 : 1 )
            if not hit:
                tree( n.res, 4, rank_of(x[t], excluding m) )

        n.observe( x[t], hadMode = ( m >= 0 ), hit )
```

`tree` is an adaptive four-bit binary tree: one probability per internal node, decayed by
1/32 on every use, which is the ordinary arithmetic-coding model for a small alphabet.

Decoding is the mirror image. The two sides perform the same recursions in the same order,
which is the whole mechanism: the weights, the rate and the residue trees are not in the
stream, so a single divergence would desynchronise them without any checksum noticing.

The first visit to a node is the one place where the bit is not sent. There is no mode to
be right or wrong about, and the decoder can see the emptiness for itself, so the symbol
is coded on its own and no hit/miss bit is spent.

### 4.3 The three recursions

**The mode** is a set of sixteen weights per node. Every visit decays all sixteen by 1/32
and raises the symbol that arrived by 64, so the heaviest weight is the recent mode and a
context that changes behaviour switches mode after a few dozen positions. The constants are
chosen so a weight saturates near 2048 and a stale one falls below the noise within the
same order of visits.

**The hit/miss rate** is a leaky integrator held at 2^16:

```
p += ( PSCALE - p ) >> PSHIFT     on a hit
p -= p >> PSHIFT                  on a miss
```

Each outcome closes a fixed fraction of the gap between the estimate and the outcome,
which is the same recurrence read two ways: it is the integer form of an exponentially
weighted average, and it needs no counter to divide by.

with `PSCALE = 2^16`, `PSHIFT = 4` and the eleven-bit probability handed to the coder as
`p >> 5`, clamped to `[1, 2047]`. One 32-bit integer replaces a pair of counters, the
reciprocal table that a counted ratio needs, and the halving logic that gives such a ratio
its memory: the integrator forgets geometrically by construction, at a rate set by a single
shift, and the estimate can still reach 2047/2048, so a context that never misses is
priced at the coder's precision. `PSHIFT` is swept over the benchmark; the surface falls
monotonically from 8 to 4 and turns back up at 3, so 4 is the measured minimum.

**The residue tree** uses the same leaky rule internally, one probability per tree node,
with the shift of 5 that the coder applies to every adaptive tree.

With that, the whole system runs one estimator rather than two: an integer that closes a
fixed fraction of the gap to the outcome it has just seen. It differs only in where the
number is kept and how much precision it carries, 16 bits for the model's own confidence
and 11 bits inside the adaptive trees, which is the width the coder consumes.

None of the three constants is in the format. They can be changed without moving a bit in
the stream, because both sides rebuild the same numbers from decoded symbols.

### 4.4 The rank mapping

The excluded symbol is removed and the rest are ranked ascending, so the test reads the
*original* symbol:

```
rank = symbol - (symbol > mode)
```

and the inverse walks ascending, on the value produced so far:

```
symbol = rank; if (symbol >= mode) symbol++
```

The residue alphabet is fifteen symbols; a rank of 15 is not a symbol and is refused as a
malformed stream rather than emitted.

## 5. Vocabulary

- **symbol**: a 4-bit nibble; the byte stream is split into two per byte.
- **context**: the `d` symbols immediately before a position, most recent first.
- **answer**: the most frequent successor of a context in the whole input. It decides the
  collapse and is never transmitted.
- **mode**: the symbol a node predicts at a given position, the heaviest of its sixteen
  weights. It changes as the node sees symbols, and it is derived rather than sent.
- **collapse**: deleting a node whose answer equals its parent's answer.
- **kept node**: survives the collapse and meets the support threshold.
- **closure**: the kept nodes plus every node on a path to one; this is the transmitted
  trie.
- **support**: how many positions carry a given context.
- **hit**: the position's symbol was the mode; the hit/miss bit says so.
- **residue**: on a miss, the symbol among the fifteen that are not the mode.
- **weight**: one of a node's sixteen recency counters. Decayed by 1/32 per visit, raised
  by 64 on the symbol that arrived; the heaviest is the mode.
- **integrator**: the node's probability that its mode is right, as a single scaled integer
  moving toward the outcome it has just seen.
- **C1, C3**: the two channels the output is accounted in. **C1** is the trie, **C3** is
  the position stream. Their sum plus the 27-byte header and two 4-byte checksums is the
  file size.

## 6. Measured results

Canterbury corpus, 2,810,784 bytes, each file compressed as a standalone input at the
compiled default `S = 20`. Sizes are exact and reproducible; the MB/s figures move a few
percent with machine load.

| file | bytes | compressed | ratio |
|---|---|---|---|
| alice29.txt | 152,089 | 60,227 | 0.3960 |
| asyoulik.txt | 125,179 | 53,700 | 0.4290 |
| cp.html | 24,603 | 11,935 | 0.4851 |
| fields.c | 11,150 | 5,531 | 0.4961 |
| grammar.lsp | 3,721 | 2,107 | 0.5662 |
| kennedy.xls | 1,029,744 | 146,290 | 0.1421 |
| lcet10.txt | 426,754 | 149,267 | 0.3498 |
| plrabn12.txt | 481,861 | 184,167 | 0.3822 |
| ptt5 | 513,216 | 57,319 | 0.1117 |
| sum | 38,240 | 18,326 | 0.4792 |
| xargs.1 | 4,227 | 2,783 | 0.6584 |
| **total** | **2,810,784** | **691,652** | **0.2461** |

The support threshold is the one structural knob, and it was swept over the whole corpus
before the default was fixed:

| S | 6 | 8 | 10 | 12 | 16 | 18 | **20** | 22 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| bytes | 700,456 | 695,774 | 694,215 | 693,321 | 691,839 | 691,894 | **691,652** | 693,972 | 694,565 | 695,790 |

Throughput on the same corpus, in memory and excluding file I/O: about 6 MB/s compress and
14.5 MB/s decompress. Compression is slower because the model build sorts the whole
input; decompression is two decisions per position plus one sixteen-way scan for the mode.

| channel | bytes | share |
|---|---|---|
| C1 model | 15,855 | 2.3% |
| C3 positions | 675,412 | 97.6% |
| header and two checksums | 385 | 0.06% |

All eleven files round-trip byte-exactly. Edge cases verified: empty, 1, 2 and 3 bytes
(45, 45, 46 and 47 bytes), a 2,097,152-byte run of 0x00 (424 B), a 1,048,576-byte run of
0xFF (235 B), and an odd-length 100,001-byte file (2,325 B). Two compressions of the same
input are byte-identical.

## 7. Known limits

- **The residue has no model of its own.** Each node adapts a sixteen-slot tree from the
  residues it has seen, which is slow to learn at nodes with few misses and carries no
  information across nodes. A shared or recency-weighted residue model is the obvious next
  step and is not attempted here.
- **The mode is learned, so it starts wrong.** A node's first visit is coded without a
  hit/miss bit at all, and its early visits are predicted by a weight vector that has seen
  almost nothing. This is charged in the measured sizes, not hidden.
- **The integrator forgets fast.** A shift of 4 gives a memory of about fifteen outcomes,
  which is what the corpus prefers and what a context that changes behaviour needs. The
  price is that a long stationary run is re-learned rather than kept: on the two largest
  text files this costs a little against a longer memory, and the corpus total is the
  compromise.
- **Incompressible input expands.** There is no stored mode: the trie and the residue
  alphabet are paid for whether or not they help, so random input grows.
- **Both sides keep the same per-node state**: sixteen weights, sixteen residue-tree probabilities and one
  integrator per node, 68 bytes per node, about 1.52 MB across the 22,331 nodes of this
  corpus, in addition to the trie itself. The state is rebuilt from decoded symbols, so it
  costs memory rather than bytes.
- **`c` and `d` do not take a threshold argument.** Only `t <input> [S]` accepts one, so
  the threshold used to produce a file is always the compiled default. `t` reports the
  better of two timed repetitions, which is why its MB/s figures are slightly optimistic.
- **Size cap.** Both directions share one limit: 2^30 symbols, i.e. 512 MiB of input. The
  encoder checks the file size before allocating and refuses anything larger. The encoder
  needs about 24 bytes per input byte for the radix sort, so the cap implies roughly
  12 GiB of working set; an allocation failure is caught and reported.

## 8. Integrity

Two independent FNV-1a checksums, each verified before the region it protects is used: the
**header checksum** over the 27 header bytes (the magic is compared first, then the
checksum over all 27 before any other field is used), and the **data checksum** over the
payload, verified before any output is produced.

Beyond the checksums the decoder validates that the two section lengths equal the payload
size exactly, that the context length and the root child count are in range, that the
decoded size is within a hard bound and can be allocated, and that the rebuilt trie
matches the encoder's node count and an order-independent hash of every node's
`(depth, edge)`. The section reader records whether it read past the end of its stream, and
the decode loop checks that flag as it goes rather than only at the end. A residue rank
naming a symbol outside the alphabet is refused rather than emitted.

The declared position count is also bounded by what the stream could possibly code: the
most confident bit the coder can represent costs 0.000705 bits, so no stream of `l3` bytes
can code more than about 11,300 positions per byte, and a declared size beyond that is
rejected before any buffer is allocated. A crafted container can still ask for a large
buffer and be rejected after some work; the bound caps the amplification at roughly 12,000
times the container rather than leaving it unbounded.

**What the checksums do not do.** FNV-1a is unkeyed: it detects corruption, not tampering.
Anyone who rewrites a header field and recomputes the checksum gets a stream the decoder
accepts, and two header fields are not independently evidenced:

- **`firstNib`** is the first symbol of the decoded stream, so changing it produces a file
  whose first nibble differs and which is otherwise valid, though some values are rejected
  because the resulting stream must still decode;
- **`srcLen`** is the declared position count. Within the slack the coder allows, changing
  it changes how many positions are decoded from the same bytes, and a highly predictable
  stream can absorb many extra positions without exhausting.

Rejecting such a stream would require redundancy rather than a checksum, such as a second
copy of the size or a keyed digest. This is an accidental-corruption design: the
round-trip tests exercise every position of every file, but a deliberately re-checksummed
file is a different valid stream rather than a rejected one.

Because the weights, the integrator and the residue trees are learned rather than
transmitted, they are not covered by any checksum, since they are not in the stream. Their
correctness rests on the encoder and decoder performing the same recursions in the same
order.

## 9. References

- C. E. Shannon, "Prediction and entropy of printed English", *Bell System Technical
  Journal* 30(1), 1951: the fraction of positions a predictor gets right as the quantity
  that bounds its code, which is the accounting this design is organised around.
- E. Fredkin, "Trie memory", *Communications of the ACM* 3(9), 1960.
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, section 6.3: digital search trees
  and tries.
- D. R. Morrison, "PATRICIA: practical algorithm to retrieve information coded in
  alphanumeric", *Journal of the ACM* 15(4), 1968: path-compressed tries.
- R. G. Brown, *Statistical Forecasting for Inventory Control*, 1959: the exponentially
  weighted moving average, the leaky integrator whose integer form prices the hit/miss bit.
- P. M. McIlroy, K. Bostic, M. D. McIlroy, "Engineering radix sort", *Computing Systems*
  6(1), 1993: the stable counting-sort passes used for the context keys.
- U. Manber, G. Myers, "Suffix arrays: a new method for on-line string searches", *SIAM
  Journal on Computing* 22(5), 1993: ordering positions by their context.
- J. Rissanen, "Modeling by shortest data description", *Automatica* 14(5), 1978: the MDL
  view behind support pruning, and behind the question of whether a statistic is worth
  transmitting or worth learning from traffic.
- L. D. Davisson, "Universal noiseless coding", *IEEE Transactions on Information Theory*
  19(6), 1973: estimating a source's statistics online, which is what the learned rates and
  weights do.
- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., 2006, ch. 13:
  universal coding and the redundancy of sequential estimation.
- I. H. Witten, R. M. Neal, J. G. Cleary, "Arithmetic coding for data compression",
  *Communications of the ACM* 30(6), 1987: the range coder used for both channels.
- G. Fowler, L. C. Noll, K.-P. Vo, "Fowler/Noll/Vo hash", 1991: the FNV-1a checksums.
