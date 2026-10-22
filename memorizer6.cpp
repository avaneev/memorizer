/**
 * @file memorizer6.cpp
 *
 * @brief Memoriser compressor.
 *
 * E-mail: aleksey.vaneev@gmail.com or info@voxengo.com
 *
 * @section license License
 *
 * Copyright (c) 2026 Aleksey Vaneev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// ---------------------------------------------------------------------------
// memorizer6 -- a memoriser corpus compressor that keeps its own reliability
// statistics by counting, never by writing them down.
//
//   memorizer6 c <in> <out>    compress
//   memorizer6 d <in> <out>    decompress
//   memorizer6 t <in>          sizes, channels, ratio and a round-trip check
//
// THE METHOD. The input is re-blocked into 4-bit symbols (nibbles). Every distinct
// context of length 1..Q over that stream is a node of a trie; a node's ANSWER is
// the symbol that most often follows its context (ties break to the smallest
// symbol). Two reductions turn that trie into a compact model:
//
//   Mode-preserving collapse: a node whose answer equals its parent's answer is
//   dropped, because a position whose deepest match is dropped gets the same
//   answer from its nearest surviving ancestor. No prediction changes.
//
//   Support pruning: a node costs model bytes to state while a repaired miss
//   costs bits to repair, so a context seen fewer than S times is not worth shipping.
//
// KEPT versus CLOSURE. These are different sets, and the distinction is the whole
// construction:
//   kept(n)    = n survives the collapse and its support is at least S;
//   closure(n) = kept(n), or n lies on a path from the root to a kept node.
// A closure node is pure structure: it carries no answer of its own, only the
// "inherit" code, and exists so the decoder can reach a kept descendant.
//
// WHAT A NODE STATES AND WHAT IT LEARNS. The model states two things per kept node
// and learns two others:
//
//   stated   the answer, and a fallback -- the symbol that most often replaced the
//            answer when it was wrong;
//   learned  how often the answer has been right lately, and how often a miss has
//            been the fallback lately.
//
// The learned pair is not transmitted. Both sides count the outcomes as they code,
// so the decoder reconstructs each node's reliability from the positions it has
// already decoded -- it learns the model's confidence in the model, at the model's
// own pace. A position is coded as: was it the answer, was it the fallback, and
// failing both, which of the fourteen remaining symbols. There is no flag channel
// and no residual channel, because the decision they would carry is either stated
// by the node or learned by counting what the node did.
//
// A rate that counted every outcome since the file began would also go deaf: a
// context whose answer was right nine times in ten, long ago, would still be trusted
// at nine in ten now, and its counts would stop moving once they grew large. The
// counts therefore hold a fixed memory of a few dozen outcomes, halved when they
// fill, so recent evidence dominates and old evidence decays by a factor of two each
// time the cap is reached. That memory length is a modelling choice with a large
// effect on the position channel; COUNT_LIMIT below states it and the README carries
// the sweep that chose it.
//
// TWO CHANNELS, charged separately. They sum to the file minus the 32-byte header and
// the two 4-byte checksums:
//   C1 model      the closure of the memoriser: edges, answers, child counts, fallbacks
//                 and profile indices, DFS pre-order
//   C3 positions  the three-way outcome at every position
//
// CONSTRUCTION. No pointer trie and no hash table build the model: one radix sort
// of the packed Q-symbol keys orders the positions, and every shorter depth is read
// off that same order by shifting the key, because positions sharing a (d)-suffix
// are a contiguous run of the (d+1)-ordered array. Counts, modal answers and
// supports of every context of every length come from one sort and Q linear scans;
// the collapse's parent test is a binary search over the next-shorter depth's
// already-sorted group keys. The closure is emitted in one DFS pass. The fallback
// symbols come from one further walk of the input over the finished closure.
//
// COMPILE
//   g++ -O3 -s -std=c++11 -pedantic-errors -Wall -Wextra -o memorizer6.exe memorizer6.cpp
//
//   -O3 is worth its keep here: the coding path is a long run of inlined
//   probabilities and shifts, and the optimiser finds a few per cent in it. -s only
//   strips symbols and changes no behaviour.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// What a node learns about itself.
//
// Two Bernoulli rates decide the coding of every position that lands on a node: how
// often the answer is right, and -- among the misses -- how often the missing symbol
// is the fallback. Neither is transmitted. Both sides count the outcomes as they go
// and code the next position at the running estimate, so the counts must be updated
// identically by the encoder and the decoder.
//
// A rate built by counting every outcome since the file began would treat an answer
// that was right nine times in ten, a million positions ago, as nine-in-ten reliable
// now, and it would stop moving altogether once the counts grew large. The counts
// therefore have a fixed memory: at COUNT_LIMIT both are halved and counting goes on,
// so recent outcomes dominate and older ones decay by a factor of two each time the
// cap is reached. The ratio survives the halving, the magnitude does not, and the
// reciprocal table stays small enough to sit in cache.
//
// COUNT_LIMIT is the length of that memory, and it is worth as much as any other
// design choice here: shortening it from thousands to tens is the single largest
// modelling gain available, because most of the predictability in real inputs is
// local. The value is swept over the benchmark in the README; the surface is flat
// between 48 and 88 and 80 is the measured minimum. No format byte depends on it,
// since both sides rebuild the counts from the symbols alone.
static const std::uint32_t COUNT_LIMIT = 80;

// One exception, and it is not a detail: a short memory cannot express near-certainty.
// Halving a counter that is right every single time caps the estimate at roughly
// COUNT_LIMIT / (COUNT_LIMIT + 1) = 0.988, which costs a hundred times more per
// position than the 0.9995 the same counter expresses when it is allowed to keep
// counting. A stream of zeros would pay that on every position. So a node that has
// never yet been wrong keeps counting to HARD_LIMIT, which restores the full range of
// confidence; its first miss drops it to COUNT_LIMIT, and there it stays, because from
// then on the useful question is what the node has been doing lately. The two limits
// cost one comparison between them.
static const std::uint32_t HARD_LIMIT = 4095;

struct learned
{
	std::uint32_t hit = 0;      // outcomes where the prediction was right
	std::uint32_t span = 0;     // outcomes seen

	// floor(2^32 / (2*span + 2)) for every span a capped counter can hold, built once
	static const std::vector< std::uint32_t >& recip()
	{
		static std::vector< std::uint32_t > T;

		if( T.empty() )
		{
			T.reserve( (size_t) HARD_LIMIT + 1 );

			for( std::uint32_t s = 0; s <= HARD_LIMIT; s++ )
			{
				const std::uint64_t den = 2ull * (std::uint64_t) s + 2ull;
				T.push_back( (std::uint32_t) ( ( (std::uint64_t) 1 << 32 ) / den ) );
			}
		}

		return( T );
	}

	// The Krichevsky-Trofimov estimate (hits + 1/2) / (span + 1) as an 11-bit
	// probability: 2048 * (2*hit + 1) / (2*span + 2). The reciprocal is tabulated and
	// multiplied in, so no division appears on the coding path; the shift of 21 is
	// 32 - 11, the width the coder wants.
	int prob() const
	{
		const std::uint64_t num = 2ull * (std::uint64_t) hit + 1ull;
		std::uint64_t v = ( num * (std::uint64_t) recip()[ (size_t) span ]
			+ ( 1ull << 20 ) ) >> 21;

		if( v < 1 ) v = 1;
		if( v > 2047 ) v = 2047;

		return( (int) v );
	}

	// Record one outcome. When the count reaches COUNT_LIMIT both counts are halved, so
	// the estimate keeps a fixed memory. Both tested conditions live inside that rare
	// branch, so the path every position takes is one comparison and one increment.
	void observe( bool right )
	{
		hit += right ? 1u : 0u;

		if( ++span >= COUNT_LIMIT )
		{
			if( hit < span || span >= HARD_LIMIT )
			{
				hit = ( hit + 1 ) >> 1;
				span = ( span + 1 ) >> 1;
			}
		}
	}
};

// How many shared tail profiles the nodes are clustered into. A profile is a
// distribution over the fourteen symbols left after the answer and the fallback are
// excluded; each node names one with a 4-bit index, so sixteen is the natural width.
static const int PROFILES = 16;
static const int TAIL = 15;

struct params
{
	int Q = 8;                 // memoriser context length in symbols
	int S = 15;                // support threshold, swept on the benchmark
};

// ---------------------------------------------------------------------------
// A carryless binary range coder with adaptive 11-bit probabilities. It codes one
// bit at a time, and values through a bit tree. Both sides update the
// probabilities identically, so no probability table is transmitted.
struct rcEnc
{
	std::uint64_t low;
	std::uint32_t range;
	unsigned char cache;
	long long cacheSize;
	std::vector< unsigned char > out;

	rcEnc() : low( 0 ), range( 0xFFFFFFFFu ), cache( 0 ), cacheSize( 1 ) {}

	void shiftLow()
	{
		if( (std::uint32_t) ( low >> 32 ) != 0 || low < 0xFF000000ull )
		{
			const unsigned char carry = (unsigned char) ( low >> 32 );

			do
			{
				out.push_back( (unsigned char) ( cache + carry ) );
				cache = 0xFF;
			} while( --cacheSize );

			cache = (unsigned char) ( low >> 24 );
		}

		cacheSize++;
		low = ( low << 8 ) & 0xFFFFFFFFull;
	}

	void bit( std::uint16_t& prob, int b )
	{
		const std::uint32_t bound = ( range >> 11 ) * (std::uint32_t) prob;

		if( b == 0 )
		{
			range = bound;
			prob = (std::uint16_t) ( prob + ( ( 2048 - prob ) >> 5 ) );
		}
		else
		{
			low += bound;
			range -= bound;
			prob = (std::uint16_t) ( prob - ( prob >> 5 ) );
		}

		while( range < ( 1u << 24 ) )
		{
			range <<= 8;
			shiftLow();
		}
	}

	// one binary decision at a FIXED probability supplied by the caller. The model
	// carries the probability, so nothing is learned here and nothing is transmitted
	// per position beyond the decision itself. prob is 11-bit: p0 = prob / 2048.
	void bitP( std::uint16_t prob, int b )
	{
		if( prob < 1 ) prob = 1;
		if( prob > 2047 ) prob = 2047;

		const std::uint32_t bound = ( range >> 11 ) * (std::uint32_t) prob;

		if( b == 0 )
		{
			range = bound;
		}
		else
		{
			low += bound;
			range -= bound;
		}

		while( range < ( 1u << 24 ) )
		{
			range <<= 8;
			shiftLow();
		}
	}

	void tree( std::uint16_t* p, int bits, int v )
	{
		int m = 1;

		for( int i = bits - 1; i >= 0; i-- )		{
			const int b = ( v >> i ) & 1;
			bit( p[ m ], b );
			m = ( m << 1 ) | b;
		}
	}

	void flush()
	{
		for( int i = 0; i < 5; i++ )
		{
			shiftLow();
		}
	}
};

struct rcDec
{
	std::uint32_t range, code;
	const unsigned char* q;
	size_t n, pos;
	bool over;          // set when the stream is read past its end

	void init( const unsigned char* p, size_t len )
	{
		q = p;
		n = len;
		pos = 0;
		range = 0xFFFFFFFFu;
		code = 0;
		over = false;

		for( int i = 0; i < 5; i++ )
		{
			code = ( code << 8 ) | get();
		}
	}

	unsigned char get()
	{
		if( pos < n )
		{
			return( q[ pos++ ] );
		}

		over = true;
		return( 0 );
	}

	int bit( std::uint16_t& prob )
	{
		const std::uint32_t bound = ( range >> 11 ) * (std::uint32_t) prob;
		int b;

		if( code < bound )
		{
			range = bound;
			prob = (std::uint16_t) ( prob + ( ( 2048 - prob ) >> 5 ) );
			b = 0;
		}
		else
		{
			code -= bound;
			range -= bound;
			prob = (std::uint16_t) ( prob - ( prob >> 5 ) );
			b = 1;
		}

		while( range < ( 1u << 24 ) )
		{
			range <<= 8;
			code = ( code << 8 ) | get();
		}

		return( b );
	}

	// the decoder's half of bitP: the same fixed probability, no update
	int bitPD( std::uint16_t prob )
	{
		if( prob < 1 ) prob = 1;
		if( prob > 2047 ) prob = 2047;

		const std::uint32_t bound = ( range >> 11 ) * (std::uint32_t) prob;
		int b;

		if( code < bound )
		{
			range = bound;
			b = 0;
		}
		else
		{
			code -= bound;
			range -= bound;
			b = 1;
		}

		while( range < ( 1u << 24 ) )
		{
			range <<= 8;
			code = ( code << 8 ) | get();
		}

		return( b );
	}

	int tree( std::uint16_t* p, int bits )
	{
		int m = 1;

		for( int i = 0; i < bits; i++ )
		{
			m = ( m << 1 ) | bit( p[ m ] );
		}

		return( m - ( 1 << bits ) );
	}
};

// ---------------------------------------------------------------------------
// one distinct (d)-symbol context, taken from the sorted order
struct group
{
	std::uint32_t key;      // the d symbols, most recent in the high nibble
	int support;            // positions carrying this context
	unsigned char mode;     // modal successor
};

// the closure as a trie; node 0 is the root
struct closure
{
	std::vector< unsigned char > edge;
	std::vector< int > ans;                   // -1 = inherit the parent's answer
	std::vector< std::vector< int > > ch;
	std::vector< int > depth;

	// Per-node conditional statistics. These are what makes the flag and the
	// residual unnecessary as separate channels: instead of one global "was the
	// model right" bit and one global "what was the symbol really" rank, every node
	// carries its own answer-scope, so a position is coded with the model of the
	// node it landed on. Both rates are memorised by the two sides as they go.
	// the rate of the fallback symbol among the node's misses, and fbSym is that
	// fallback (0..15, or -1 when the node never misses).
	std::vector< int > fbSym;

	// Both rates are memorised by the two sides as they go; the model states the
	// answer and the fallback symbol and nothing about its own reliability.
	std::vector< learned > rate;      // answer rate
	std::vector< learned > fbRate;    // fallback rate, over the node's misses

	// Which shared tail profile codes this node's residual tail. The profile itself
	// is not stored here: the coder keeps one adaptive tree per profile, and the index
	// is all the stream carries.
	std::vector< unsigned char > prof;

	// fill every node's fallback and tail profile from the input; defined below the
	// walk it needs
	void learn( const std::vector< unsigned char >& x, int M, int Q, int rootMode );

	void reset()
	{
		edge.assign( 1, 0 );
		ans.assign( 1, -1 );
		ch.assign( 1, std::vector< int >() );
		depth.assign( 1, 0 );
		fbSym.assign( 1, -1 );
		rate.assign( 1, learned() );
		fbRate.assign( 1, learned() );
		prof.assign( 1, 0 );
	}

	int addNode( int parent, int e )
	{
		edge.push_back( (unsigned char) e );
		ans.push_back( -1 );
		ch.push_back( std::vector< int >() );
		depth.push_back( depth[ (size_t) parent ] + 1 );
		fbSym.push_back( -1 );
		rate.push_back( learned() );
		fbRate.push_back( learned() );
		prof.push_back( 0 );
		const int id = (int) edge.size() - 1;
		ch[ (size_t) parent ].push_back( id );
		return( id );
	}

	void sortChildren()
	{
		for( size_t i = 0; i < ch.size(); i++ )
		{
			std::sort( ch[ i ].begin(), ch[ i ].end(), [ & ]( int a, int b )
			{
				return( edge[ (size_t) a ] < edge[ (size_t) b ] );
			} );
		}
	}

	long long keptCount() const
	{
		long long n = 0;

		for( size_t i = 0; i < ans.size(); i++ )
		{
			n += ( ans[ i ] >= 0 ) ? 1 : 0;
		}

		return( n );
	}

	// order-independent sum of per-node FNV-1a values: the encoder walks insertion
	// order and the decoder DFS order, so a chained hash would differ
	std::uint32_t keptHash() const
	{
		std::uint32_t acc = 0;

		for( size_t i = 0; i < ans.size(); i++ )
		{
			if( ans[ i ] < 0 )
			{
				continue;
			}

			std::uint32_t h = 2166136261u;
			h = ( h ^ (std::uint32_t) depth[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) edge[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) ans[ i ] ) * 16777619u;
			acc += h;
		}

		return( acc );
	}

	// DFS pre-order: edge | answer code (0..15 answer, 16 inherit) | child count |
	// fallback symbol, 0..16 on the wire: 0 means the node has no fallback, 1..16 name
	// rates are counted by the two sides rather than transmitted.
	void ser( rcEnc& e, std::uint16_t* pe, std::uint16_t* pa, std::uint16_t* pc,
		std::uint16_t* ps, std::uint16_t* pp, int node ) const
	{
		const int a = ans[ (size_t) node ];

		e.tree( pe, 4, (int) edge[ (size_t) node ] );
		e.tree( pa, 5, ( a >= 0 ) ? a : 16 );
		e.tree( pc, 5, (int) ch[ (size_t) node ].size() );

		if( a >= 0 )
		{
			e.tree( ps, 5, fbSym[ (size_t) node ] + 1 );
			e.tree( pp, 4, (int) prof[ (size_t) node ] );
		}

		for( size_t c = 0; c < ch[ (size_t) node ].size(); c++ )
		{
			ser( e, pe, pa, pc, ps, pp, ch[ (size_t) node ][ c ] );
		}
	}
};

// the decoder's view: the same shape, filled by the parser
struct closureView
{
	std::vector< unsigned char > edge;
	std::vector< int > ans;
	std::vector< std::vector< int > > ch;
	std::vector< int > depth;
	std::vector< int > fbSym;
	std::vector< learned > rate;
	std::vector< learned > fbRate;
	std::vector< unsigned char > prof;

	// Rebuild one node and its descendants from the model stream, appending to this
	// view and returning the node index, or -1 if the stream is malformed. Only the
	// decoder calls it, so it belongs to the structure it fills.
	int parse( rcDec& d, std::uint16_t* pe, std::uint16_t* pa, std::uint16_t* pc,
		std::uint16_t* ps, std::uint16_t* pp, int parentDepth, int Q, long long maxNodes );
};

int closureView::parse( rcDec& d, std::uint16_t* pe, std::uint16_t* pa, std::uint16_t* pc,
	std::uint16_t* ps, std::uint16_t* pp, int parentDepth, int Q, long long maxNodes )
{
	if( parentDepth + 1 > Q || (long long) edge.size() >= maxNodes )
	{
		return( -1 );
	}

	const int e = d.tree( pe, 4 );
	const int code = d.tree( pa, 5 );
	const int nc = d.tree( pc, 5 );

	if( code < 0 || code > 16 || nc < 0 || nc > 16 )
	{
		return( -1 );
	}

	// a kept node states a fallback and a tail profile; a structure-only node carries
	// the inherit code and nothing else
	int fs = -1;
	int pf = 0;

	if( code != 16 )
	{
		fs = d.tree( ps, 5 ) - 1;      // 0..16 on the wire, -1 means "no fallback"

		if( fs > 15 )
		{
			return( -1 );
		}

		pf = d.tree( pp, 4 );

		if( pf < 0 || pf >= PROFILES )
		{
			return( -1 );
		}
	}

	const int id = (int) edge.size();
	edge.push_back( (unsigned char) e );
	ans.push_back( ( code == 16 ) ? -1 : code );
	ch.push_back( std::vector< int >() );
	depth.push_back( parentDepth + 1 );
	fbSym.push_back( fs );
	rate.push_back( learned() );
	fbRate.push_back( learned() );
	prof.push_back( (unsigned char) pf );

	for( int i = 0; i < nc; i++ )
	{
		const int c = parse( d, pe, pa, pc, ps, pp, parentDepth + 1, Q, maxNodes );

		if( c < 0 )
		{
			return( -1 );
		}

		ch[ (size_t) id ].push_back( c );
	}

	return( id );
}


// ---------------------------------------------------------------------------
// The walk that turns a position into the node whose answer will be used.
//
// descend by the context symbol, returning the node whose answer was used
// (node 0, the root, when no closure node carried an answer)
template< class TR >
static inline int walkNode( const TR& V, const std::vector< unsigned char >& x, int t, int Q,
	int rootMode, int& pred )
{
	int node = 0;
	int last = rootMode;
	int lastNode = 0;

	for( int d = 1; d <= Q && d <= t; d++ )
	{
		const std::vector< int >& C = V.ch[ (size_t) node ];
		int next = -1;

		for( size_t i = 0; i < C.size(); i++ )
		{
			if( (int) V.edge[ (size_t) C[ i ] ] == (int) x[ (size_t) ( t - d ) ] )
			{
				next = C[ i ];
				break;
			}
		}

		if( next < 0 )
		{
			break;
		}

		node = next;

		if( V.ans[ (size_t) node ] >= 0 )
		{
			last = V.ans[ (size_t) node ];
			lastNode = node;
		}
	}

	pred = last;
	return( lastNode );
}

// ---------------------------------------------------------------------------
// What each node learns from the input.
//
// Two things are settled here, both encoder-side, because both are properties of the
// whole input rather than of a single position:
//
//   the fallback -- the symbol that most often replaces the answer when the answer is
//   wrong. It has to be known before a position is coded, so it is transmitted.
//
//   the tail profile -- which of the PROFILES shared distributions best describes the
//   residual tail at this node. Only the 4-bit index travels; the distributions
//   themselves are adaptive trees in the coder, learned as the positions arrive.
//
// The clustering is a k-means over the nodes' tail histograms with a cross-entropy
// objective: a profile is judged by how cheaply it codes the tails given to it.
// Initialisation is farthest-first from the heaviest node, so the result is
// deterministic and needs no random seed.
void closure::learn( const std::vector< unsigned char >& x, int M, int Q, int rootMode )
{
	const size_t N = ans.size();
	std::vector< long long > vis( N, 0 );
	std::vector< std::vector< long long > > miss( N, std::vector< long long >( 16, 0 ) );

	for( int t = 1; t < M; t++ )
	{
		int pred = 0;
		const size_t node = (size_t) walkNode( *this, x, t, Q, rootMode, pred );
		vis[ node ]++;

		if( pred == (int) x[ (size_t) t ] )
		{
			continue;
		}

		const int tr = (int) x[ (size_t) t ];
		miss[ node ][ (size_t) ( ( tr < pred ) ? tr : tr - 1 ) ]++;
	}

	// the fallback, and the tail that remains once it is removed
	std::vector< std::vector< double > > tail( N, std::vector< double >( (size_t) TAIL, 0.0 ) );
	std::vector< double > mass( N, 0.0 );

	for( size_t i = 0; i < N; i++ )
	{
		prof[ i ] = 0;
		fbSym[ i ] = -1;

		if( vis[ i ] == 0 ) continue;

		int fb = -1;
		long long m = 0;

		for( int r = 0; r < 16; r++ )
		{
			if( miss[ i ][ (size_t) r ] > m ) { m = miss[ i ][ (size_t) r ]; fb = r; }
		}

		const int pred = ( i == 0 ) ? rootMode : ans[ i ];
		int drop = -1;

		if( fb >= 0 && m > 0 && pred >= 0 )
		{
			fbSym[ i ] = ( fb < pred ) ? fb : fb + 1;
			drop = fb;
		}

		for( int r = 0; r < 16; r++ )
		{
			if( r == drop ) continue;

			// with no fallback nothing is removed, so the rank is unchanged
			const int r2 = ( drop < 0 || r < drop ) ? r : r - 1;

			if( r2 < 0 || r2 >= TAIL ) continue;
			tail[ i ][ (size_t) r2 ] = (double) miss[ i ][ (size_t) r ];
			mass[ i ] += (double) miss[ i ][ (size_t) r ];
		}
	}

	// ---- k-means over the tail histograms ------------------------------------
	std::vector< size_t > used;

	for( size_t i = 0; i < N; i++ )
	{
		if( mass[ i ] > 0.0 ) used.push_back( i );
	}

	if( used.empty() ) return;

	std::vector< std::vector< double > > cent( (size_t) PROFILES, std::vector< double >( (size_t) TAIL, 0.0 ) );
	std::vector< int > seed;

	{
		size_t best = used[ 0 ];

		for( size_t k = 1; k < used.size(); k++ )
		{
			if( mass[ used[ k ] ] > mass[ best ] ) best = used[ k ];
		}

		seed.push_back( (int) best );

		while( (int) seed.size() < PROFILES )
		{
			double worst = -1.0;
			int pick = -1;

			for( size_t k = 0; k < used.size(); k++ )
			{
				const size_t u = used[ k ];
				double bestCost = 1e300;

				for( size_t c = 0; c < seed.size(); c++ )
				{
					const size_t s = (size_t) seed[ c ];
					double cost = 0.0;

					for( int r = 0; r < TAIL; r++ )
					{
						const double cnt = tail[ u ][ (size_t) r ];

						if( cnt > 0.0 )
						{
							cost += cnt * std::log2( std::max( 1e-9, mass[ s ] )
								/ std::max( 1e-9, tail[ s ][ (size_t) r ] ) );
						}
					}

					if( cost < bestCost ) bestCost = cost;
				}

				if( bestCost > worst ) { worst = bestCost; pick = (int) u; }
			}

			if( pick < 0 ) break;
			seed.push_back( pick );
		}
	}

	for( int c = 0; c < PROFILES; c++ )
	{
		const size_t s = (size_t) seed[ (size_t) ( ( c < (int) seed.size() ) ? c : 0 ) ];

		for( int r = 0; r < TAIL; r++ )
		{
			cent[ (size_t) c ][ (size_t) r ] = ( tail[ s ][ (size_t) r ] + 0.5 )
				/ ( mass[ s ] + 0.5 * (double) TAIL );
		}
	}

	for( int it = 0; it < 12; it++ )
	{
		std::vector< std::vector< double > > acc( (size_t) PROFILES, std::vector< double >( (size_t) TAIL, 0.0 ) );
		std::vector< double > tot( (size_t) PROFILES, 0.0 );

		for( size_t k = 0; k < used.size(); k++ )
		{
			const size_t u = used[ k ];
			double bestCost = 1e300;
			int best = 0;

			for( int c = 0; c < PROFILES; c++ )
			{
				double cost = 0.0;

				for( int r = 0; r < TAIL; r++ )
				{
					const double cnt = tail[ u ][ (size_t) r ];

					if( cnt > 0.0 ) { cost -= cnt * std::log2( cent[ (size_t) c ][ (size_t) r ] ); }
				}

				if( cost < bestCost ) { bestCost = cost; best = c; }
			}

			prof[ u ] = (unsigned char) best;
			tot[ (size_t) best ] += mass[ u ];

			for( int r = 0; r < TAIL; r++ ) acc[ (size_t) best ][ (size_t) r ] += tail[ u ][ (size_t) r ];
		}

		for( int c = 0; c < PROFILES; c++ )
		{
			if( tot[ (size_t) c ] <= 0.0 ) continue;

			for( int r = 0; r < TAIL; r++ )
			{
				cent[ (size_t) c ][ (size_t) r ] = ( acc[ (size_t) c ][ (size_t) r ] + 0.5 )
					/ ( tot[ (size_t) c ] + 0.5 * (double) TAIL );
			}
		}
	}

	// The root is not serialised with the closure, so its profile is not in the
	// stream: both sides fix it at zero. Forcing it after the clustering keeps the
	// encoder and the decoder on the same shared tree.
	prof[ 0 ] = 0;
}
// ---------------------------------------------------------------------------
// the sorted-order construction
struct model
{
	std::vector< std::vector< group > > g;
	int rootMode = 0;
	closure cl;
	long long kept = 0;

	void run( const std::vector< unsigned char >& x, int M, const params& P )
	{
		const int Q = P.Q;
		g.assign( (size_t) ( Q + 1 ), std::vector< group >() );

		std::vector< std::uint32_t > key( (size_t) M, 0 );

		for( int t = Q; t < M; t++ )
		{
			std::uint32_t k = 0;

			for( int j = 1; j <= Q; j++ )
			{
				k = ( k << 4 ) | (std::uint32_t) x[ (size_t) ( t - j ) ];
			}

			key[ (size_t) t ] = k;
		}

		std::vector< int > idx;

		for( int t = Q; t < M; t++ )
		{
			idx.push_back( t );
		}

		{
			std::vector< int > tmp( idx.size(), 0 );

			for( int pass = 0; pass < ( 4 * Q + 7 ) / 8; pass++ )
			{
				std::vector< long long > cnt( 256, 0 );
				const int shift = pass * 8;

				for( size_t i = 0; i < idx.size(); i++ )
				{
					cnt[ (size_t) ( ( key[ (size_t) idx[ i ] ] >> shift ) & 255 ) ]++;
				}

				long long sum = 0;

				for( int b = 0; b < 256; b++ )
				{
					const long long c = cnt[ (size_t) b ];
					cnt[ (size_t) b ] = sum;
					sum += c;
				}

				for( size_t i = 0; i < idx.size(); i++ )
				{
					const int t = idx[ i ];
					const int b = (int) ( ( key[ (size_t) t ] >> shift ) & 255 );
					tmp[ (size_t) cnt[ (size_t) b ]++ ] = t;
				}

				idx.swap( tmp );
			}
		}

		// every depth's groups, off the same sorted order
		for( int d = Q; d >= 1; d-- )
		{
			const int shift = 4 * ( Q - d );
			const std::uint32_t mask = ( d >= 8 ) ? 0xFFFFFFFFu
				: ( ( 1u << ( 4 * d ) ) - 1u );
			std::vector< group >& G = g[ (size_t) d ];

			for( size_t i = 0; i < idx.size(); )
			{
				const std::uint32_t k = ( key[ (size_t) idx[ i ] ] >> shift ) & mask;
				size_t j = i;
				long long c[ 16 ];

				for( int a = 0; a < 16; a++ )
				{
					c[ a ] = 0;
				}

				while( j < idx.size()
					&& ( ( key[ (size_t) idx[ j ] ] >> shift ) & mask ) == k )
				{
					c[ x[ (size_t) idx[ j ] ] ]++;
					j++;
				}

				int b = 0;

				for( int a = 1; a < 16; a++ )
				{
					if( c[ a ] > c[ b ] )
					{
						b = a;
					}
				}

				group gr;
				gr.key = k;
				gr.support = (int) ( j - i );
				gr.mode = (unsigned char) b;
				G.push_back( gr );
				i = j;
			}
		}

		// the root's answer: the global modal symbol
		{
			long long c[ 16 ];

			for( int a = 0; a < 16; a++ )
			{
				c[ a ] = 0;
			}

			for( int t = 0; t < M; t++ )
			{
				c[ x[ (size_t) t ] ]++;
			}

			for( int a = 1; a < 16; a++ )
			{
				if( c[ a ] > c[ rootMode ] )
				{
					rootMode = a;
				}
			}
		}

		// kept(n) = survives the collapse AND support >= S
		std::vector< std::vector< char > > keptFlag( (size_t) ( Q + 1 ) );

		for( int d = 1; d <= Q; d++ )
		{
			std::vector< group >& G = g[ (size_t) d ];
			keptFlag[ (size_t) d ].assign( G.size(), 0 );

			for( size_t i = 0; i < G.size(); i++ )
			{
				int parentAns = rootMode;

				if( d > 1 )
				{
					const std::uint32_t pk = G[ i ].key >> 4;
					const std::vector< group >& PG = g[ (size_t) ( d - 1 ) ];
					size_t a = 0, b = PG.size();

					while( a < b )
					{
						const size_t mid = ( a + b ) / 2;

						if( PG[ mid ].key < pk )
						{
							a = mid + 1;
						}
						else
						{
							b = mid;
						}
					}

					if( a < PG.size() && PG[ a ].key == pk )
					{
						parentAns = (int) PG[ a ].mode;
					}
				}

				keptFlag[ (size_t) d ][ i ] = ( ( (int) G[ i ].mode != parentAns )
					&& G[ i ].support >= P.S ) ? 1 : 0;
			}
		}

		// the closure: the trie built from the kept nodes supplies the path nodes
		cl.reset();

		for( int d = 1; d <= Q; d++ )
		{
			std::vector< group >& G = g[ (size_t) d ];

			for( size_t i = 0; i < G.size(); i++ )
			{
				if( !keptFlag[ (size_t) d ][ i ] )
				{
					continue;
				}

				int node = 0;

				for( int j = 0; j < d; j++ )
				{
					const int s = (int) ( ( G[ i ].key >> ( 4 * ( d - 1 - j ) ) ) & 15 );
					int next = -1;

					for( size_t c = 0; c < cl.ch[ (size_t) node ].size(); c++ )
					{
						if( cl.edge[ (size_t) cl.ch[ (size_t) node ][ c ] ] == s )
						{
							next = cl.ch[ (size_t) node ][ c ];
							break;
						}
					}

					if( next < 0 )
					{
						next = cl.addNode( node, s );
					}

					node = next;
				}

				cl.ans[ (size_t) node ] = (int) G[ i ].mode;
			}
		}

		cl.sortChildren();
		kept = cl.keptCount();
	}
};

// ---------------------------------------------------------------------------
static void putU32( std::vector< unsigned char >& v, std::uint32_t x )
{
	for( int i = 0; i < 4; i++ )
	{
		v.push_back( (unsigned char) ( ( x >> ( 8 * i ) ) & 255 ) );
	}
}

static std::uint32_t getU32( const std::vector< unsigned char >& v, size_t& p )
{
	std::uint32_t x = 0;

	for( int i = 0; i < 4; i++ )
	{
		x |= (std::uint32_t) v[ p + (size_t) i ] << ( 8 * i );
	}

	p += 4;
	return( x );
}

static std::uint32_t fnv( const unsigned char* p, size_t n )
{
	std::uint32_t ck = 2166136261u;

	for( size_t i = 0; i < n; i++ )
	{
		ck = ( ck ^ (std::uint32_t) p[ i ] ) * 16777619u;
	}

	return( ck );
}

// Read a whole file, refusing anything larger than the caller's cap BEFORE the
// buffer is allocated: the size comes from the filesystem, so an oversized input is
// reported rather than turned into an allocation attempt that cannot succeed.
static std::vector< unsigned char > readFile( const char* path, bool& ok,
	std::uint64_t maxBytes, const char* what )
{
	ok = false;
	std::ifstream in( path, std::ios::binary | std::ios::ate );

	if( !in )
	{
		fprintf( stderr, "memorizer6: cannot open %s\n", path );
		return( std::vector< unsigned char >() );
	}

	const std::streamoff n = in.tellg();

	if( n < 0 )
	{
		return( std::vector< unsigned char >() );
	}

	if( (std::uint64_t) n > maxBytes )
	{
		fprintf( stderr, "memorizer6: %s exceeds the size limit\n", what );
		return( std::vector< unsigned char >() );
	}

	in.seekg( 0 );

	try
	{
		std::vector< unsigned char > v( (size_t) n );

		if( n > 0 )
		{
			in.read( (char*) v.data(), (std::streamsize) n );

			if( !in )
			{
				return( std::vector< unsigned char >() );
			}
		}

		ok = true;
		return( v );
	}
	catch( const std::exception& e )
	{
		fprintf( stderr, "memorizer6: cannot allocate the input buffer\n" );
		return( std::vector< unsigned char >() );
	}
}

static bool writeFile( const char* path, const std::vector< unsigned char >& v )
{
	std::ofstream o( path, std::ios::binary );

	if( !o )
	{
		return( false );
	}

	if( !v.empty() )
	{
		o.write( (const char*) v.data(), (std::streamsize) v.size() );
	}

	return( !!o );
}

// header, 32 bytes: magic 4 | srcLen 4 | Q, firstNib, rootMode, rootNch 4 | kept 4 |
// kept hash 4 | section lengths 8 | root hit, fb, fbSym, reserved 4
static const size_t HEADER = 32;

// the largest input either side will handle, in symbols: 2^30 nibbles = 512 MiB of
// bytes. One constant for both directions, so the encoder can never produce a stream
// the decoder refuses.
static const std::uint64_t MAX_POSITIONS = (std::uint64_t) 1 << 30;

// ---------------------------------------------------------------------------
struct codec
{
	std::vector< unsigned char > out;
	long long c1 = 0, c3 = 0;
};

static bool encode( const std::vector< unsigned char >& src, const params& P, codec& C )
{
	if( (std::uint64_t) src.size() * 2 > MAX_POSITIONS )
	{
		fprintf( stderr, "memorizer6: input too large\n" );
		return( false );
	}

	const int M = (int) src.size() * 2;
	std::vector< unsigned char > x( (size_t) M );

	for( size_t i = 0; i < src.size(); i++ )
	{
		x[ 2 * i ] = (unsigned char) ( ( src[ i ] >> 4 ) & 15 );
		x[ 2 * i + 1 ] = (unsigned char) ( src[ i ] & 15 );
	}

	model mo;
	mo.cl.reset();

	if( M > P.Q )
	{
		mo.run( x, M, P );
	}
	else
	{
		// too short for any context: the model is a bare root whose answer is the
		// global modal symbol, which is what the decoder will derive from the header
		long long c[ 16 ];

		for( int a = 0; a < 16; a++ )
		{
			c[ a ] = 0;
		}

		for( int t = 0; t < M; t++ )
		{
			c[ x[ (size_t) t ] ]++;
		}

		for( int a = 1; a < 16; a++ )
		{
			if( c[ a ] > c[ mo.rootMode ] )
			{
				mo.rootMode = a;
			}
		}
	}

	// the per-node conditional statistics: this is the step that replaces the
	// global flag and residual channels. Every node learns its own answer rate and
	// its own fallback, so a position is coded by the model of the node it hits.
	mo.cl.learn( x, M, P.Q, mo.rootMode );

	std::vector< unsigned char > m1, ps;
	{
		rcEnc e;
		std::uint16_t pe[ 16 ], pa[ 32 ], pc[ 32 ], pss[ 32 ], pp[ 16 ];

		for( int i = 0; i < 16; i++ ) pe[ i ] = 1024;
		for( int i = 0; i < 32; i++ ) pa[ i ] = pc[ i ] = 1024;
		for( int i = 0; i < 32; i++ ) pss[ i ] = 1024;
		for( int i = 0; i < 16; i++ ) pp[ i ] = 1024;

		for( size_t c = 0; c < mo.cl.ch[ 0 ].size(); c++ )
		{
			mo.cl.ser( e, pe, pa, pc, pss, pp, mo.cl.ch[ 0 ][ c ] );
		}

		e.flush();
		m1 = e.out;
	}

	// one position channel. Each position asks the node it landed on three
	// questions: was it the node's answer (a fixed probability from the model), was
	// it the node's fallback (a second fixed probability), and failing both, which of
	// the fourteen remaining symbols (an adaptive tree, since that tail is the same
	// everywhere). There is no separate hit/miss flag and no separate residual.
	{
		rcEnc e;
		// one adaptive tail tree per profile: nodes that share a profile also share
		// what is learned about their tails, which is the whole point of the index
		std::uint16_t ptail[ PROFILES ][ 16 ];

		for( int i = 0; i < PROFILES; i++ )
		{
			for( int j = 0; j < 16; j++ ) ptail[ i ][ j ] = 1024;
		}

		for( int t = 1; t < M; t++ )
		{
			int pred = 0;
			const size_t node = (size_t) walkNode( mo.cl, x, t, P.Q, mo.rootMode, pred );
			const int tr = (int) x[ (size_t) t ];
			const int ans = ( node == 0 ) ? mo.rootMode : mo.cl.ans[ node ];
			const int fb = mo.cl.fbSym[ node ];
			const int p0 = mo.cl.rate[ node ].prob();
			const int p1 = mo.cl.fbRate[ node ].prob();

			if( tr == ans )
			{
				e.bitP( (std::uint16_t) p0, 0 );
			}
			else
			{
				e.bitP( (std::uint16_t) p0, 1 );

				if( fb >= 0 && tr == fb )
				{
					e.bitP( (std::uint16_t) p1, 0 );
				}
				else
				{
					// the fallback bit exists only when the node has one, so that the
					// decoder never reads a bit the encoder did not write
					if( fb >= 0 )
					{
						e.bitP( (std::uint16_t) p1, 1 );
					}

					// the tail: a rank among the fourteen symbols that are neither
					// the answer nor the fallback
					int skip1 = ans;
					int skip2 = ( fb >= 0 ) ? fb : -1;

					// ascending, so the two removals can be undone in a defined order
					if( skip2 >= 0 && skip2 < skip1 ) { const int z = skip1; skip1 = skip2; skip2 = z; }

					// the rank counts how many of the removed symbols lie below the
					// true one, so both tests read the ORIGINAL symbol
					int rk = tr;

					if( tr > skip1 ) rk--;
					if( skip2 >= 0 && tr > skip2 ) rk--;

					e.tree( ptail[ mo.cl.prof[ node ] ], 4, rk );
				}
			}

			// both rates move forward with every position, so the next visit to this
			// node is coded at a better estimate; the fallback rate is learned over the
			// misses, and only where a fallback applies
			mo.cl.rate[ node ].observe( tr == ans );

			if( tr != ans && fb >= 0 ) { mo.cl.fbRate[ node ].observe( tr == fb ); }
		}

		e.flush();
		ps = e.out;
	}

	C.c1 = (long long) m1.size();
	C.c3 = (long long) ps.size();

	C.out.clear();
	C.out.push_back( 'm' );
	C.out.push_back( 'e' );
	C.out.push_back( 'm' );
	C.out.push_back( '1' );
	putU32( C.out, (std::uint32_t) src.size() );
	C.out.push_back( (unsigned char) P.Q );
	C.out.push_back( (unsigned char) ( ( M > 0 ) ? x[ 0 ] : 0 ) );
	C.out.push_back( (unsigned char) mo.rootMode );
	C.out.push_back( (unsigned char) mo.cl.ch[ 0 ].size() );
	putU32( C.out, (std::uint32_t) mo.kept );
	putU32( C.out, mo.cl.keptHash() );
	putU32( C.out, (std::uint32_t) m1.size() );
	putU32( C.out, (std::uint32_t) ps.size() );
	// the root carries statistics too: it is the node every position that matched no
	// closure node lands on, and it is not serialised with the closure
	C.out.push_back( (unsigned char) ( mo.cl.fbSym[ 0 ] + 1 ) );
	C.out.push_back( 0 );
	C.out.push_back( 0 );
	C.out.push_back( 0 );                    // reserved: keeps the header at 32 bytes
	putU32( C.out, fnv( C.out.data(), C.out.size() ) );

	const size_t payloadStart = C.out.size();

	C.out.insert( C.out.end(), m1.begin(), m1.end() );
	C.out.insert( C.out.end(), ps.begin(), ps.end() );

	putU32( C.out, fnv( C.out.data() + payloadStart, C.out.size() - payloadStart ) );
	return( true );
}

static bool decode( const std::vector< unsigned char >& src, std::vector< unsigned char >& dst )
{
	if( src.size() < HEADER + 8 || src[ 0 ] != 'm' || src[ 1 ] != 'e' || src[ 2 ] != 'm'
		|| src[ 3 ] != '1' )
	{
		fprintf( stderr, "memorizer6: not a stream or truncated header\n" );
		return( false );
	}

	// checksum 1: the header, verified before any field other than the magic is used
	std::uint32_t hdrStored = 0;

	for( int b = 0; b < 4; b++ )
	{
		hdrStored |= (std::uint32_t) src[ HEADER + (size_t) b ] << ( 8 * b );
	}

	if( fnv( src.data(), HEADER ) != hdrStored )
	{
		fprintf( stderr, "memorizer6: header checksum mismatch\n" );
		return( false );
	}

	const size_t payloadStart = HEADER + 4;
	const size_t bodyEnd = src.size() - 4;

	// checksum 2: the payload, verified before any output is produced
	std::uint32_t dataStored = 0;

	for( int b = 0; b < 4; b++ )
	{
		dataStored |= (std::uint32_t) src[ bodyEnd + (size_t) b ] << ( 8 * b );
	}

	if( fnv( src.data() + payloadStart, bodyEnd - payloadStart ) != dataStored )
	{
		fprintf( stderr, "memorizer6: data checksum mismatch\n" );
		return( false );
	}

	size_t p = 4;
	const std::uint32_t srcLen = getU32( src, p );
	const int Q = src[ p++ ];
	// the codec is defined for one context length and one symbol width; a stream
	// that declares anything else is not one we produced
	if( Q != 8 )
	{
		fprintf( stderr, "memorizer6: unsupported context length\n" );
		return( false );
	}

	const unsigned char firstNib = src[ p++ ];

	if( firstNib > 15 )
	{
		fprintf( stderr, "memorizer6: first symbol is not a nibble\n" );
		return( false );
	}
	const unsigned char rootMode = src[ p++ ];

	if( rootMode > 15 )
	{
		fprintf( stderr, "memorizer6: root answer is not a symbol\n" );
		return( false );
	}
	const int rootNch = src[ p++ ];

	// the root has at most one child per symbol
	if( rootNch > 16 )
	{
		fprintf( stderr, "memorizer6: root child count out of range\n" );
		return( false );
	}
	const std::uint32_t keptEnc = getU32( src, p );
	const std::uint32_t hashEnc = getU32( src, p );
	const std::uint32_t l1 = getU32( src, p );
	const std::uint32_t l3 = getU32( src, p );

	// the root's own statistics travel in the header: the root is not serialised
	// with the closure, but every position that matched no closure node lands on it
	const int rootFbSym = (int) src[ p++ ] - 1;

	if( rootFbSym > 15 )
	{
		fprintf( stderr, "memorizer6: root statistics out of range\n" );
		return( false );
	}

	if( (std::uint64_t) l1 + l3 != (std::uint64_t) ( bodyEnd - payloadStart ) )
	{
		fprintf( stderr, "memorizer6: section lengths do not match the file\n" );
		return( false );
	}

	const size_t o1 = payloadStart;
	const size_t o3 = o1 + l1;

	closureView V;
	V.edge.push_back( 0 );
	V.ans.push_back( -1 );
	V.ch.push_back( std::vector< int >() );
	V.depth.push_back( 0 );
	V.fbSym.push_back( rootFbSym );
	V.rate.push_back( learned() );
	V.fbRate.push_back( learned() );
	V.prof.push_back( 0 );        // the root states no profile: it never reaches a tail
	{
		rcDec d;
		d.init( src.data() + o1, l1 );
		std::uint16_t pe[ 16 ], pa[ 32 ], pc[ 32 ], pss[ 32 ], pp[ 16 ];

		for( int i = 0; i < 16; i++ ) pe[ i ] = 1024;
		for( int i = 0; i < 32; i++ ) pa[ i ] = pc[ i ] = pss[ i ] = 1024;
		for( int i = 0; i < 16; i++ ) pp[ i ] = 1024;

		for( int i = 0; i < rootNch; i++ )
		{
			const int c = V.parse( d, pe, pa, pc, pss, pp, 0, Q,
				8 * (long long) l1 + 8 );

			if( c < 0 )
			{
				fprintf( stderr, "memorizer6: malformed closure\n" );
				return( false );
			}

			V.ch[ 0 ].push_back( c );
		}

		if( d.over )
		{
			fprintf( stderr, "memorizer6: model stream exhausted\n" );
			return( false );
		}
	}

	// the rebuilt closure must be the one the encoder shipped
	{
		long long nk = 0;
		std::uint32_t acc = 0;

		for( size_t i = 0; i < V.ans.size(); i++ )
		{
			if( V.ans[ i ] < 0 )
			{
				continue;
			}

			nk++;
			std::uint32_t h = 2166136261u;
			h = ( h ^ (std::uint32_t) V.depth[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) V.edge[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) V.ans[ i ] ) * 16777619u;
			acc += h;
		}

		if( nk != (long long) keptEnc || acc != hashEnc )
		{
			fprintf( stderr, "memorizer6: rebuild mismatch: kept %lld vs %u\n", nk, keptEnc );
			return( false );
		}
	}

	// the decoded size is attacker-controlled: bound it in 64-bit arithmetic before
	// it can overflow or ask for an impossible allocation
	const std::uint64_t m64 = (std::uint64_t) srcLen * 2;

	if( m64 > MAX_POSITIONS )
	{
		fprintf( stderr, "memorizer6: declared size is out of range\n" );
		return( false );
	}

	const int M = (int) m64;

	// A position can never cost less than the most confident bit the coder can
	// represent: at p = 2047/2048 the answer costs 0.000705 bits. So a stream of l3
	// bytes can code at most 8*l3/0.000705, about 11,300 positions per byte, and a
	// declared size beyond that is impossible rather than merely unlikely. Checking it
	// here keeps a forty-byte forgery from asking for a gigabyte of buffer.
	if( M > 1 && (std::uint64_t) ( M - 1 ) > (std::uint64_t) l3 * 12000ull )
	{
		fprintf( stderr, "memorizer6: declared size does not match the coded streams\n" );
		return( false );
	}

	std::vector< unsigned char > x;

	try
	{
		x.assign( (size_t) M, 0 );
	}
	catch( const std::exception& e )
	{
		fprintf( stderr, "memorizer6: cannot allocate the decoded buffer\n" );
		return( false );
	}

	if( M > 0 ) x[ 0 ] = firstNib;

	rcDec d3;
	d3.init( src.data() + o3, l3 );
	std::uint16_t ptail[ PROFILES ][ 16 ];

	for( int i = 0; i < PROFILES; i++ )
	{
		for( int j = 0; j < 16; j++ ) ptail[ i ][ j ] = 1024;
	}

	// the mirror of the encoder: the node the position lands on supplies both
	// probabilities, so no hit/miss flag and no residual are read separately
	for( int t = 1; t < M; t++ )
	{
		// A crafted size can ask for far more positions than the stream codes. The
		// reader flags exhaustion the moment it happens, so check periodically rather
		// than only after the loop: the work then stops within a few thousand
		// positions of the stream running dry, instead of running to whatever the
		// header happened to declare.
		if( ( t & 4095 ) == 0 && d3.over )
		{
			fprintf( stderr, "memorizer6: coded stream exhausted\n" );
			return( false );
		}

		int pred = 0;
		const size_t node = (size_t) walkNode( V, x, t, Q, (int) rootMode, pred );
		const int ans = ( node == 0 ) ? (int) rootMode : V.ans[ node ];
		const int fb = V.fbSym[ node ];
		const int p0 = V.rate[ node ].prob();
		const int p1 = V.fbRate[ node ].prob();

		if( d3.bitPD( (std::uint16_t) p0 ) == 0 )
		{
			x[ (size_t) t ] = (unsigned char) ans;
		}
		else if( fb >= 0 && d3.bitPD( (std::uint16_t) p1 ) == 0 )
		{
			x[ (size_t) t ] = (unsigned char) fb;
		}
		else
		{
			const int rk = d3.tree( ptail[ V.prof[ node ] ], 4 );

			// re-insert the two removed symbols
			int skip1 = ans;
			int skip2 = ( fb >= 0 ) ? fb : -1;

			if( skip2 >= 0 && skip2 < skip1 ) { const int z = skip1; skip1 = skip2; skip2 = z; }

			// the inverse of the encoder: ascending order, each test on the value
			// produced so far
			int tr = rk;

			if( tr >= skip1 ) tr++;
			if( skip2 >= 0 && tr >= skip2 ) tr++;

			// an authored stream can name a rank the encoder never produces; the
			// symbol it would select is not a nibble, so refuse it rather than
			// emitting a byte the input could not have contained
			if( tr > 15 )
			{
				fprintf( stderr, "memorizer6: tail rank out of range\n" );
				return( false );
			}

			x[ (size_t) t ] = (unsigned char) tr;
		}

		// the same two updates the encoder made, in the same order
		const int got = (int) x[ (size_t) t ];
		V.rate[ node ].observe( got == ans );

		if( got != ans && fb >= 0 ) { V.fbRate[ node ].observe( got == fb ); }
	}

	if( d3.over )
	{
		fprintf( stderr, "memorizer6: coded stream exhausted\n" );
		return( false );
	}

	// An under-declared size decodes fewer positions than the stream encodes, which
	// leaves the section reader short of its declared length. The only slack a correct
	// stream leaves is the coder's five flush bytes, so anything beyond that is a
	// stream whose declared size does not match what it codes.
	if( l3 > d3.pos + 16 )
	{
		fprintf( stderr, "memorizer6: declared size does not match the coded streams\n" );
		return( false );
	}

	try
	{
		dst.assign( (size_t) srcLen, 0 );
	}
	catch( const std::exception& e )
	{
		fprintf( stderr, "memorizer6: cannot allocate the decoded buffer\n" );
		return( false );
	}

	for( size_t i = 0; i < dst.size(); i++ )
	{
		dst[ i ] = (unsigned char) ( ( x[ 2 * i ] << 4 ) | ( x[ 2 * i + 1 ] & 15 ) );
	}

	return( true );
}

// ---------------------------------------------------------------------------
int main( int argc, char** argv )
{
	if( argc < 3 )
	{
		fprintf( stderr, "usage: memorizer6 c|d|t <in> [out]  |  memorizer6 t <in> [S]\n" );
		return( 2 );
	}

	const std::string modeStr = argv[ 1 ];

	if( modeStr != "c" && modeStr != "d" && modeStr != "t" )
	{
		fprintf( stderr, "usage: memorizer6 c|d|t <in> [out]  |  memorizer6 t <in> [S]\n" );
		return( 2 );
	}

	const char mode = modeStr[ 0 ];

	if( argc < 4 && mode != 't' )
	{
		fprintf( stderr, "usage: memorizer6 c|d|t <in> [out]  |  memorizer6 t <in> [S]\n" );
		return( 2 );
	}

	bool ok = false;
	// an input larger than the format can represent is refused before any buffer is
	// allocated; a container is bounded by the same cap plus room for the framing
	const bool isDecode = ( mode == 'd' );
	std::vector< unsigned char > src = isDecode
		? readFile( argv[ 2 ], ok, MAX_POSITIONS + ( (std::uint64_t) 1 << 20 ), "container" )
		: readFile( argv[ 2 ], ok, MAX_POSITIONS / 2, "input" );

	if( !ok )
	{
		return( 2 );
	}

	// One support threshold for every input. It was swept over the whole corpus; see the
	// README for the table. The surface is flat across a wide range, so the value is not
	// finely tuned - it sits on a plateau rather than at a sharp optimum.
	params P;

	// `t` takes an optional support threshold so the pruning can be swept without
	// changing the format: memorizer6 t <in> [S]
	if( mode == 't' && argc >= 4 )
	{
		const int sv = atoi( argv[ 3 ] );

		if( sv < 1 || sv > 255 )
		{
			fprintf( stderr, "memorizer6: support threshold must be 1..255\n" );
			return( 2 );
		}

		P.S = sv;
	}

	if( mode == 'c' )
	{
		codec C;
		bool okEnc = false;

		// the model build allocates several vectors the size of the input, so an
		// allocation failure has to be a message rather than std::terminate
		try
		{
			okEnc = encode( src, P, C );
		}
		catch( const std::exception& e )
		{
			fprintf( stderr, "memorizer6: cannot allocate the model\n" );
			return( 2 );
		}

		if( !okEnc )
		{
			return( 2 );
		}

		if( !writeFile( argv[ 3 ], C.out ) )
		{
			fprintf( stderr, "cannot write %s\n", argv[ 3 ] );
			return( 2 );
		}

		const long long sum = C.c1 + C.c3 + (long long) HEADER + 8;

		printf( "channels: C1 model %lld | C3 positions %lld | header %lld |"
			" checksum 8 | sum %lld = file %d\n", C.c1, C.c3, (long long) HEADER,
			sum, (int) C.out.size() );

		if( sum != (long long) C.out.size() )
		{
			fprintf( stderr, "memorizer6: CHANNEL SUM MISMATCH\n" );
			return( 2 );
		}

		printf( "compressed %d -> %d bytes (%.4fx)\n", (int) src.size(), (int) C.out.size(),
			(double) C.out.size() / (double) std::max< size_t >( 1, src.size() ) );
		return( 0 );
	}

	if( mode == 'd' )
	{
		std::vector< unsigned char > out;
		bool okDec = false;

		try
		{
			okDec = decode( src, out );
		}
		catch( const std::exception& e )
		{
			fprintf( stderr, "memorizer6: cannot allocate the model\n" );
			return( 2 );
		}

		if( !okDec )
		{
			return( 2 );
		}

		if( !writeFile( argv[ 3 ], out ) )
		{
			fprintf( stderr, "cannot write %s\n", argv[ 3 ] );
			return( 2 );
		}

		printf( "decompressed %d -> %d bytes\n", (int) src.size(), (int) out.size() );
		return( 0 );
	}

	// t: compress and decompress in memory, timed, and report throughput. File I/O is
	// outside both timings. Two passes each, keeping the faster, which removes most of
	// the first-touch cost of the allocator.
	codec C;
	double bestC = 1e30, bestD = 1e30;
	bool same = false;

	for( int rep = 0; rep < 2; rep++ )
	{
		const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

		try
		{
			if( !encode( src, P, C ) )
			{
				return( 2 );
			}
		}
		catch( const std::exception& e )
		{
			fprintf( stderr, "memorizer6: cannot allocate the model\n" );
			return( 2 );
		}

		const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::vector< unsigned char > back;
		bool okDec = false;

		try
		{
			okDec = decode( C.out, back );
		}
		catch( const std::exception& e )
		{
			fprintf( stderr, "memorizer6: cannot allocate the model\n" );
			return( 2 );
		}

		const std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

		const double cs = std::chrono::duration< double >( t1 - t0 ).count();
		const double ds = std::chrono::duration< double >( t2 - t1 ).count();

		if( cs < bestC ) bestC = cs;
		if( ds < bestD ) bestD = ds;

		same = okDec && ( back.size() == src.size() )
			&& ( src.empty() || std::memcmp( back.data(), src.data(), src.size() ) == 0 );
	}

	const double mb = (double) src.size() / 1000000.0;
	const double cMBs = ( bestC > 0.0 ) ? mb / bestC : 0.0;
	const double dMBs = ( bestD > 0.0 ) ? mb / bestD : 0.0;
	const long long sum = C.c1 + C.c3 + (long long) HEADER + 8;

	// an empty input has no ratio and no throughput to report
	if( src.empty() )
	{
		printf( "input 0 B | compressed %d B | ratio n/a | MB/s n/a | round-trip %s\n",
			(int) C.out.size(), same ? "OK" : "FAILED" );
	}
	else
	{
		printf( "input %d B | compressed %d B | ratio %.4f | %.2f MB/s compress |"
			" %.2f MB/s decompress | round-trip %s\n", (int) src.size(), (int) C.out.size(),
			(double) C.out.size() / (double) std::max< size_t >( 1, src.size() ), cMBs, dMBs,
			same ? "OK" : "FAILED" );
	}
	printf( "channels: C1 model %lld | C3 positions %lld | header %lld | checksum 8"
		" | sum %lld = file %d\n", C.c1, C.c3, (long long) HEADER, sum,
		(int) C.out.size() );

	if( sum != (long long) C.out.size() )
	{
		fprintf( stderr, "memorizer6: CHANNEL SUM MISMATCH\n" );
		return( 2 );
	}

	return( same ? 0 : 1 );
}
