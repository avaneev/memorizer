# compress4a

A corpus compressor built from a **memoriser** of context tries plus a
correction stream, with a model built by a single radix sort.

```
compress4a c <input> <output>    compress
compress4a d <input> <output>    decompress
compress4a t <input>             size, ratio, throughput, channels, round-trip check
```

C++11, ASCII-only source, in-memory I/O, no external libraries.

## 1. The idea

Compression here means *transmit a model that reproduces the input, then transmit a
correction for every position the model gets wrong*. There is no held-out set and no
generalisation objective: the model is built from the whole input and the decoder must
reproduce that input byte for byte.

The model is a trie of contexts over a 4-bit symbol stream:

- the input bytes are re-blocked into **nibbles** (4-bit symbols), so a 500 KB file
  becomes one million positions;
- every distinct context of length 1..Q (Q = 8) is a node;
- a node's **answer** is the symbol that most often follows that context over the whole
  input, ties breaking to the smallest symbol.

Two reductions make that trie small enough to transmit.

**Mode-preserving collapse.** A node whose answer equals its parent's answer is dropped.
A position whose deepest matching node is dropped therefore receives the same answer from
its nearest surviving ancestor, so no prediction changes — the saving is free. On English
text roughly 85% of nodes disappear this way.

**Support pruning.** A node costs about 8.7 bits to transmit and a repaired miss about
2.9 bits, so a context seen fewer than S times cannot pay for itself. The threshold is
S = 6, chosen by sweeping it over the benchmark (section 4.1).

**Kept versus closure.** These are different sets, and separating them is what makes the
model both compact and decodable:

```
kept(n)    = n survives the collapse and support(n) >= S
closure(n) = kept(n), or n lies on a path from the root to a kept node
```

A closure node that is not kept is pure structure: it carries no answer, only an
"inherit" code, and exists so the decoder can walk to a kept descendant. The transmitted
set is the closure, about 1.6x the kept count on this corpus. The answer field is a single
code with an escape — `0..15` is an answer, `16` means "inherit" — not two flags.

**No corrector stage.** Positions the memoriser misses are carried by the residual channel
alone, and channels C2 and C4 are structurally zero. Adding a corrector would change the
format: the header carries three section lengths and no model2 or flag2 field. The
design treats a corrector's value as conditional on the model leaving a concentrated miss
population, so the choice between having one and not having one has to be measured per
input by encoding both ways; **that measurement has not been made here**, and the absence of
the stage should be read as the baseline this build establishes, not as a tested optimum.

## 2. The algorithm

### 2.1 Construction by one radix sort

The classical way to build such a trie is a pointer tree with a hash lookup per position
per depth. This implementation does neither: it orders the positions once and reads every
context length off that order.

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
            node = descend(closure, g.key's d symbols)   # creating nodes as needed
            node.answer = g.mode
```

The cost is one radix sort plus Q linear scans for every context count of every length,
and one binary search per group for the collapse.

### 2.2 Coding

Every channel goes through one **adaptive binary range coder** with 11-bit probabilities.
No probability table is transmitted, because both sides update identically after every bit.

```
encode_model(closure):          # DFS pre-order, children in ascending symbol order
    per node: edge (4-bit tree) | answer code (5-bit tree, 16 = inherit)
              | child count (5-bit tree, 0..16)

encode_positions:
    for t in 1..M-1:
        pred = walk(closure, context of t)      # deepest answer seen
        flag1.append(pred == x[t])
        if pred != x[t]:
            residual.append(rank of x[t] among the 15 symbols excluding pred)
```

The residual is coded as a **rank** rather than a symbol. The true symbol is never the
prediction, so the coded alphabet is the fifteen symbols that can actually occur; the
prediction is re-inserted at decode time. This is worth about 0.006 of ratio on the
benchmark, and it costs nothing at either end.

### 2.3 Channels

Five channels are charged separately; together with the two 4-byte checksums and the
32-byte header they sum to the file size, which the program asserts:

| channel | holds | paid per |
|---|---|---|
| C1 model1 | the closure, DFS pre-order | once |
| C2 model2 | the corrector's closure (zero here) | once |
| C3 flag1 | hit or miss | position |
| C4 flag2 | corrected or not (zero here) | miss |
| C5 residual | the rank of the true symbol at a miss | miss |

### 2.4 Integrity

Two independent FNV-1a checksums, each verified before the region it protects is used:

- **header checksum** over the 32 header bytes, verified before any field is read, so a
  corrupted length or context parameter cannot be acted on;
- **data checksum** over the payload, verified before any output is produced.

Beyond the checksums the decoder validates that the three section lengths equal the
payload size exactly, that the declared context length is the one the format defines, that
the decoded size is within a hard bound and can be allocated, and that the rebuilt closure
matches the encoder's kept count and an order-independent hash of every kept node's
(depth, edge, answer). Each range decoder also records whether it read past the end of its
section.

## 3. Vocabulary

- **symbol** — a 4-bit nibble; the byte stream is split into two per byte.
- **context** — the `d` symbols immediately before a position, most recent first.
- **answer / mode** — the most frequent successor of a context over the input.
- **collapse** — deleting a node whose answer equals its parent's answer.
- **kept node** — survives the collapse and meets the support threshold.
- **closure** — the kept nodes plus every node on a path to one; the extra nodes carry the
  inherit code.
- **support** — how many positions carry a given context.
- **flag1** — one bit per position: did the model's answer match.
- **rank** — the position of the true symbol in the symbol order with the prediction
  removed.
- **channel** — an independently charged part of the output.

## 4. Measured results

Canterbury corpus, 2,810,784 bytes, each file compressed as a standalone input. Throughput
is measured in memory, excluding file I/O, as the better of two passes.

| file | bytes | compressed | ratio | MB/s compress | MB/s decompress |
|---|---|---|---|---|---|
| alice29.txt | 152,089 | 68,973 | 0.4535 | 4.92 | 12.40 |
| asyoulik.txt | 125,179 | 62,424 | 0.4987 | 4.63 | 11.70 |
| cp.html | 24,603 | 12,276 | 0.4990 | 4.03 | 15.50 |
| fields.c | 11,150 | 5,353 | 0.4801 | 4.65 | 17.81 |
| grammar.lsp | 3,721 | 2,123 | 0.5705 | 3.61 | 16.18 |
| kennedy.xls | 1,029,744 | 316,329 | 0.3072 | 9.59 | 44.22 |
| lcet10.txt | 426,754 | 176,977 | 0.4147 | 4.70 | 10.93 |
| plrabn12.txt | 481,861 | 230,919 | 0.4792 | 4.35 | 10.52 |
| ptt5 | 513,216 | 65,462 | 0.1276 | 10.68 | 41.00 |
| sum | 38,240 | 18,879 | 0.4937 | 4.09 | 17.32 |
| xargs.1 | 4,227 | 2,718 | 0.6430 | 3.08 | 13.86 |
| **total** | **2,810,784** | **962,433** | **0.3424** | **6.46** | **18.87** |

The aggregate throughput is the corpus size divided by the sum of the per-file times.

All eleven files round-trip byte-exactly. Edge cases verified: empty, 1 byte, 2 bytes,
3 bytes, all-0x00, all-0xFF, and a 100,001-byte file at odd nibble parity.

### 4.1 The support threshold

`S` was swept over the corpus in {1, 2, 3, 4, 6, 8, 12}:

| S | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| total bytes | 1,038,339 | 984,981 | 967,194 | 960,709 | **957,592** | 959,075 | 964,731 |

The totals are for the nine files of 11 KB and larger, where the threshold is free to act.
`S = 6` is the best single value; `S = 4` is 3,117 bytes behind and every individual text
file has its optimum at 4 or 6, so one value serves all of them. Very small `S` is clearly
worse (`S = 1` costs 8%) because the model then carries once-seen contexts.

The one file that prefers a much higher threshold is `ptt5`, 16-bit audio: its best is
`S = 12`. Audio and binary data have less exploitable repetition, so a node is worth less
there. A per-input rule was not adopted: the gain is 1,135 bytes on the whole corpus and it
would need a data-type classifier whose own threshold would be fitted to this benchmark.

### 4.2 Where the bits go

For `alice29.txt` the channels split as model 10.9%, flag1 43.9%, residual 45.2%. Across
the corpus the flag and the residual are 93.95% of the output (465,766 + 438,444 of 962,433): the model is already
small, and almost all of the cost is the model telling the decoder whether it was right.
That is why the productive next step is the miss population rather than a larger model.

### 4.3 Known limits

The adaptive coder's probability state is 11 bits with a shift of 5, so a context
saturates at p = 2017/2048. On ordinary data this is never reached, but on long runs of a
single symbol it leaves a floor of about 0.022 bits per bit. A 2 MB run of zeros therefore
costs about 11 KB where a coder with a wider state would spend a few hundred bytes. The
benchmark contains no such input, so the ratio is unaffected; a build aimed at highly
redundant data should widen the state or add an escape at extreme probability.
- **Size cap.** Both directions share one limit: 2^30 symbols, i.e. 512 MiB of input.
  The encoder refuses anything larger, so it cannot produce a stream its own decoder
  would reject.
- **The under-declaration check has a tolerance.** It compares each section reader's
  position against its declared length and allows 16 bytes of slack for the encoder's
  flush, so a declared size short by fewer than about 17 bytes is still accepted. The
  check catches gross under-declaration, not a single missing symbol.
- **A crafted size is bounded but not free.** A stream declaring 2^29 symbols allocates
  1 GB and takes about 6 seconds before it is rejected. The allocation is capped and
  wrapped, so it fails cleanly rather than crashing, but a hostile stream can still cost
  that much time.

## 5. References

- E. Fredkin, "Trie memory", *Communications of the ACM* 3(9), 1960.
- D. E. Knuth, *The Art of Computer Programming*, vol. 3, section 6.3: digital search
  trees and tries.
- D. R. Morrison, "PATRICIA -- practical algorithm to retrieve information coded in
  alphanumeric", *Journal of the ACM* 15(4), 1968: path-compressed tries.
- P. M. McIlroy, K. Bostic, M. D. McIlroy, "Engineering radix sort", *Computing Systems*
  6(1), 1993: the stable counting-sort passes used for the context keys.
- U. Manber, G. Myers, "Suffix arrays: a new method for on-line string searches", *SIAM
  Journal on Computing* 22(5), 1993: ordering positions by their context instead of
  storing contexts.
- J. Rissanen, "Modeling by shortest data description", *Automatica* 14(5), 1978: the MDL
  view behind support pruning, where a node is kept only if it saves more than it costs.
- I. H. Witten, R. M. Neal, J. G. Cleary, "Arithmetic coding for data compression",
  *Communications of the ACM* 30(6), 1987: the range coder used for every channel.
- G. Fowler, L. C. Noll, K.-P. Vo, "Fowler/Noll/Vo hash", 1991: the FNV-1a checksums.
