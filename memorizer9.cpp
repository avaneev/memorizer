/**
 * @file memorizer9.cpp
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
// memorizer9 -- a memoriser that transmits a trie of contexts and nothing else, and
// codes the input as a hit/miss bit per position plus a residue symbol on a miss.
//
//   memorizer9 c <in> <out>    compress
//   memorizer9 d <in> <out>    decompress
//   memorizer9 t <in> [S]      sizes, channels, ratio, throughput, round-trip check
//
// THE METHOD. The input is re-blocked into 4-bit symbols (nibbles). Every distinct
// context of length 1..Q over that stream is a node of a trie, and the node's ANSWER is
// the symbol that most often follows its context (ties break to the smallest symbol).
// Two reductions turn that trie into a small model: a node whose answer equals its
// parent is dropped, which shrinks the model. The collapse is a size reduction, not a
// semantic one: predictions come from each node's own learned weights, so a dropped node
// sends its positions to an ancestor whose predictor differs, and the benchmark measures
// what that costs. A context seen fewer than S times is dropped, because a stated node
// costs model bytes while the misses it repairs are already cheap.
//
// THREE RECURSIONS, TWO CHANNELS. The stream carries the shape of the trie; the coding
// carries two decisions per position:
//
//   the hit/miss bit   was the symbol the one this node predicted?
//   the residue        on a miss, which of the other fifteen symbols was it?
//
// The prediction itself is never transmitted. Each node keeps sixteen successor weights,
// decayed by 1/32 per visit, and its MODE is the heaviest of them, so the predictor
// follows what the context has been doing lately rather than what it did on average.
// The decoder rebuilds those weights from the symbols it has already decoded, so the same
// prediction is available at both ends without a byte being spent on it.
//
// WHAT THE MODEL CHANNEL CARRIES. Only the trie: one 4-bit edge and one 5-bit child
// count per node, in DFS pre-order. On this corpus that is about 5.6 bits per node. The
// modes, the hit rates and the residue statistics are not in the stream at all.
//
// HOW THE ESTIMATES MOVE. Three integer recursions, none of them a ratio of counters,
// a table or a division:
//
//   weights   w -= w >> 5, then w[symbol] += 64, so the mode follows recent behaviour;
//   rate      p += (PSCALE - p) >> 4 on a hit and p -= p >> 4 on a miss, held at 2^16
//             and handed to the coder as eleven bits;
//   residue   the ordinary adaptive tree, which decays each node the same way.
//
// The decay constants are the whole model of non-stationarity: they decide how fast a
// context forgets, and they are the only tuning in the system. None of them is in the
// format, because both sides derive the same numbers from the symbols they have seen.
//
// TWO CHANNELS, charged separately. They sum to the file minus the 27-byte header and
// the two 4-byte checksums:
//   C1 model      the trie: edges and child counts, DFS pre-order
//   C3 positions  the hit/miss bit at every position, and the residue on every miss
//
// CONSTRUCTION. No pointer trie and no hash table build the model: one radix sort of the
// packed Q-symbol keys orders the positions, and every shorter depth is read off that
// same order by shifting the key, because positions sharing a (d)-suffix are a contiguous
// run of the (d+1)-ordered array. Counts, modal answers and supports of every context of
// every length come from one sort and Q linear scans; the collapse's parent test is a
// binary search over the next-shorter depth's already-sorted group keys. The closure is
// emitted in one DFS pass.
//
// COMPILE
//   g++ -O3 -s -std=c++11 -pedantic-errors -Wall -Wextra -o memorizer9.exe memorizer9.cpp
//
//   -O3 earns its keep here: the coding path is a long run of inlined probabilities and
//   shifts, and the optimiser finds a few per cent in it. -s only strips symbols.
// ---------------------------------------------------------------------------
#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// What a node learns about itself.
//
// One probability decides the coding of every position that lands on a node: how often
// the node mode has been right. It is not transmitted. Both sides move the same number
// by the same rule as they go, so the estimate is rebuilt from decoded symbols alone.
//
// The estimate is an integer leaky integrator rather than a counted ratio. The count and
// the ratio are one number: each outcome moves the current value toward zero or toward
// full scale by 1/2^PSHIFT of the distance that is left, so evidence decays
// geometrically, and no counter, no division and no reciprocal table appears anywhere.
// The state is one 32-bit integer and the update is a shift and an add on the coding path.
//
// PSCALE sets the precision the estimate is held at, PBITS the width handed to the coder
// and PSHIFT how fast it forgets. With PSCALE = 2^16 and PBITS = 5 the coder sees eleven
// bits, and a shift of 4 gives a memory of about fifteen outcomes, the range where a
// context's recent behaviour still dominates its history. None of the three is in the
// format: they can change without moving a bit in the stream, because both sides derive
// the estimate from the symbols they have already coded.
static const std::int32_t PSCALE = 1 << 16;
static const int PSHIFT = 4;
static const int PBITS = 5;

struct integrator
{
	std::int32_t p = PSCALE / 2;

	// one outcome: close 1/2^PSHIFT of the gap to the outcome
	void observe( bool right )
	{
		if( right ) { p += ( PSCALE - p ) >> PSHIFT; }
		else { p -= p >> PSHIFT; }
	}

	// the coder wants eleven bits; the value is at most 2047 by construction, and the
	// floor is the smallest probability the coder accepts
	int prob() const
	{
		const int v = ( int ) ( p >> PBITS );

		return( v < 1 ? 1 : v );
	}
};
// The context length is a compile-time constant: the packed key holds Q four-bit symbols
// in one 32-bit word, so a Q above 8 would shift a word by its own width. Q and S are the
// only two knobs; S is the threshold a context must reach before it is stated.
static const int Q = 8;
static_assert( Q >= 2 && Q <= 8, "Q must fit a 32-bit context key" );

struct params
{
	int S = 20;                // support threshold, swept on the benchmark
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
	// carries the probability, so nothing is integrator here and nothing is transmitted
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

// The trie, and everything a node learns about itself while the stream is coded. The model
// channel carries the shape of the trie and nothing else: no answers, no rates, no residue
// statistics. A node's prediction, and the confidence in it, are rebuilt from the symbols
// the two sides have already coded.
//
//   w     sixteen successor weights, decayed by 1/32 per visit. The node MODE is the
//         heaviest of them, so the predictor follows the input local behaviour rather
//         than its lifetime average.
//   rate  an integrator pricing the bit that says whether the mode was right.
//   res   the adaptive tree that codes which of the other fifteen symbols a miss was.
struct trie
{
	std::vector< unsigned char > edge;
	std::vector< std::vector< int > > ch;
	std::vector< int > depth;
	std::vector< integrator > rate;
	std::vector< std::int16_t > w;
	std::vector< std::uint16_t > res;

	// the deepest node whose edge path matches the context that ends at position t
	int walk( const std::vector< unsigned char >& x, int t ) const
	{
		int node = 0;

		for( int d = 1; d <= Q && d <= t; d++ )
		{
			const std::vector< int >& kids = ch[ (size_t) node ];
			int next = -1;

			for( size_t i = 0; i < kids.size(); i++ )
			{
				if( (int) edge[ (size_t) kids[ i ] ] == (int) x[ (size_t) ( t - d ) ] )
				{
					next = kids[ i ];
					break;
				}
			}

			if( next < 0 ) { break; }

			node = next;
		}

		return( node );
	}

	// the mode: the heaviest successor weight, ties to the lowest symbol, and -1 before
	// the node has seen anything
	int mode( int node ) const
	{
		const std::int16_t* r = &w[ (size_t) node * 16 ];
		int best = -1;

		for( int s = 0; s < 16; s++ )
		{
			if( r[ s ] > 0 && ( best < 0 || r[ s ] > r[ best ] ) ) { best = s; }
		}

		return( best );
	}

	// one outcome: the weights decay by 1/32 and the symbol gains, while the rate that
	// prices the hit/miss bit moves only where there was a mode to be right or wrong about
	void observe( int node, int sym, bool hadMode, bool hit )
	{
		std::int16_t* r = &w[ (size_t) node * 16 ];

		for( int s = 0; s < 16; s++ ) { r[ s ] -= ( r[ s ] + 31 ) >> 5; }

		r[ sym ] += 64;

		if( hadMode ) { rate[ (size_t) node ].observe( hit ); }
	}

	long long nodeCount() const
	{
		return( (long long) edge.size() );
	}

	// order-independent sum of per-node FNV-1a values over the shape alone: the encoder
	// walks insertion order and the decoder DFS order, so a chained hash would differ
	std::uint32_t nodeHash() const
	{
		std::uint32_t acc = 0;

		for( size_t i = 0; i < edge.size(); i++ )
		{
			std::uint32_t h = 2166136261u;
			h = ( h ^ (std::uint32_t) depth[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) edge[ i ] ) * 16777619u;
			acc += h;
		}

		return( acc );
	}

	void reset()
	{
		edge.assign( 1, 0 );
		ch.assign( 1, std::vector< int >() );
		depth.assign( 1, 0 );
		rate.assign( 1, integrator() );
		w.assign( 16, 0 );
		res.assign( 16, 1024 );
	}

	int addNode( int parent, int e )
	{
		edge.push_back( (unsigned char) e );
		ch.push_back( std::vector< int >() );
		depth.push_back( depth[ (size_t) parent ] + 1 );
		rate.push_back( integrator() );
		w.insert( w.end(), 16, 0 );
		res.insert( res.end(), 16, 1024 );
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
};

// the encoder trie: it also holds the modal successors the build computes, which decide
// the collapse and are never written to the stream
struct closure : trie
{

	// DFS pre-order: edge (4 bits) and child count (5 bits), then each child in turn
	void ser( rcEnc& e, std::uint16_t* pe, std::uint16_t* pc, int node ) const
	{
		e.tree( pe, 4, (int) edge[ (size_t) node ] );
		e.tree( pc, 5, (int) ch[ (size_t) node ].size() );

		for( size_t c = 0; c < ch[ (size_t) node ].size(); c++ )
		{
			ser( e, pe, pc, ch[ (size_t) node ][ c ] );
		}
	}
};

// the decoder trie: filled by the parser
struct closureView : trie
{
	// Rebuild one node and its descendants from the model stream, appending to this view
	// and returning the node index, or -1 if the stream is malformed.
	int parse( rcDec& d, std::uint16_t* pe, std::uint16_t* pc, int parentDepth,
		long long maxNodes );
};

int closureView::parse( rcDec& d, std::uint16_t* pe, std::uint16_t* pc, int parentDepth,
	long long maxNodes )
{
	if( parentDepth + 1 > Q || (long long) edge.size() >= maxNodes )
	{
		return( -1 );
	}

	const int e = d.tree( pe, 4 );
	const int nc = d.tree( pc, 5 );

	if( nc > 16 )
	{
		return( -1 );
	}

	const int id = (int) edge.size();

	edge.push_back( (unsigned char) e );
	ch.push_back( std::vector< int >() );
	depth.push_back( parentDepth + 1 );
	rate.push_back( integrator() );
	w.insert( w.end(), 16, 0 );
	res.insert( res.end(), 16, 1024 );

	for( int i = 0; i < nc; i++ )
	{
		const int c = parse( d, pe, pc, parentDepth + 1, maxNodes );

		if( c < 0 )
		{
			return( -1 );
		}

		ch[ (size_t) id ].push_back( c );
	}

	return( id );
}


// ---------------------------------------------------------------------------
// the sorted-order construction
struct model
{
	std::vector< std::vector< group > > g;
	int rootMode = 0;
	closure cl;

	void run( const std::vector< unsigned char >& x, int M, const params& P )
	{
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


			}
		}

		cl.sortChildren();
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
		fprintf( stderr, "memorizer9: cannot open %s\n", path );
		return( std::vector< unsigned char >() );
	}

	const std::streamoff n = in.tellg();

	if( n < 0 )
	{
		return( std::vector< unsigned char >() );
	}

	if( (std::uint64_t) n > maxBytes )
	{
		fprintf( stderr, "memorizer9: %s exceeds the size limit\n", what );
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
	catch( const std::exception& )
	{
		fprintf( stderr, "memorizer9: cannot allocate the input buffer\n" );
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

// header, 27 bytes: magic 4 | srcLen 4 | Q, firstNib, rootNch 3 | nodes 4 | node hash 4 |
// section lengths 8
static const size_t HEADER = 27;

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
		fprintf( stderr, "memorizer9: input too large\n" );
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

	if( M > Q )
	{
		mo.run( x, M, P );
	}


	std::vector< unsigned char > m1, ps;
	{
		rcEnc e;
		std::uint16_t pe[ 16 ], pc[ 32 ];

		for( int i = 0; i < 16; i++ ) { pe[ i ] = 1024; }
		for( int i = 0; i < 32; i++ ) { pc[ i ] = 1024; }

		for( size_t c = 0; c < mo.cl.ch[ 0 ].size(); c++ )
		{
			mo.cl.ser( e, pe, pc, mo.cl.ch[ 0 ][ c ] );
		}

		e.flush();
		m1 = e.out;
	}

	// One position channel, two decisions. Every position costs one hit/miss bit, priced
	// by the node it landed on. A miss then costs one symbol among the other fifteen,
	// through that node's own adaptive tree. Nothing here refers to a symbol the model
	// was told about: the mode is whatever the node's own weights currently favour, and
	// the decoder reaches the same mode from the same history.
	{
		rcEnc e;

		for( int t = 1; t < M; t++ )
		{
			const size_t node = (size_t) mo.cl.walk( x, t );
			const int tr = (int) x[ (size_t) t ];
			const int m = mo.cl.mode( (int) node );
			bool hit = false;

			if( m < 0 )
			{
				// nothing is known here yet: the symbol is coded on its own, and no
				// hit/miss bit is spent because the decoder can see the same emptiness
				e.tree( &mo.cl.res[ node * 16 ], 4, tr );
			}
			else
			{
				hit = ( tr == m );
				e.bitP( (std::uint16_t) mo.cl.rate[ node ].prob(), hit ? 0 : 1 );

				if( !hit )
				{
					// the rank counts the removed symbol, so the test reads the ORIGINAL
					// symbol and the inverse walks ascending
					int rk = tr;

					if( tr > m ) { rk--; }

					e.tree( &mo.cl.res[ node * 16 ], 4, rk );
				}
			}

			mo.cl.observe( (int) node, tr, m >= 0, hit );
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
	C.out.push_back( (unsigned char) Q );
	C.out.push_back( (unsigned char) ( ( M > 0 ) ? x[ 0 ] : 0 ) );
	C.out.push_back( (unsigned char) mo.cl.ch[ 0 ].size() );
	putU32( C.out, (std::uint32_t) mo.cl.nodeCount() );
	putU32( C.out, mo.cl.nodeHash() );
	putU32( C.out, (std::uint32_t) m1.size() );
	putU32( C.out, (std::uint32_t) ps.size() );

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
		fprintf( stderr, "memorizer9: not a stream or truncated header\n" );
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
		fprintf( stderr, "memorizer9: header checksum mismatch\n" );
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
		fprintf( stderr, "memorizer9: data checksum mismatch\n" );
		return( false );
	}

	size_t p = 4;
	const std::uint32_t srcLen = getU32( src, p );
	const int qDecl = src[ p++ ];
	// the codec is defined for one context length and one symbol width; a stream
	// that declares anything else is not one we produced
	if( qDecl != ::Q )
	{
		fprintf( stderr, "memorizer9: unsupported context length\n" );
		return( false );
	}

	const unsigned char firstNib = src[ p++ ];

	if( firstNib > 15 )
	{
		fprintf( stderr, "memorizer9: first symbol is not a nibble\n" );
		return( false );
	}
	const int rootNch = src[ p++ ];

	// the root has at most one child per symbol
	if( rootNch > 16 )
	{
		fprintf( stderr, "memorizer9: root child count out of range\n" );
		return( false );
	}
	const std::uint32_t keptEnc = getU32( src, p );
	const std::uint32_t hashEnc = getU32( src, p );
	const std::uint32_t l1 = getU32( src, p );
	const std::uint32_t l3 = getU32( src, p );



	if( (std::uint64_t) l1 + l3 != (std::uint64_t) ( bodyEnd - payloadStart ) )
	{
		fprintf( stderr, "memorizer9: section lengths do not match the file\n" );
		return( false );
	}

	const size_t o1 = payloadStart;
	const size_t o3 = o1 + l1;

	closureView V;
	V.edge.push_back( 0 );
	V.ch.push_back( std::vector< int >() );
	V.depth.push_back( 0 );
	V.rate.push_back( integrator() );
	V.w.insert( V.w.end(), 16, 0 );
	V.res.insert( V.res.end(), 16, 1024 );
	{
		rcDec d;
		d.init( src.data() + o1, l1 );
		std::uint16_t pe[ 16 ], pc[ 32 ];

		for( int i = 0; i < 16; i++ ) { pe[ i ] = 1024; }
		for( int i = 0; i < 32; i++ ) { pc[ i ] = 1024; }

		for( int i = 0; i < rootNch; i++ )
		{
			const int c = V.parse( d, pe, pc, 0, (long long) keptEnc );

			if( c < 0 )
			{
				fprintf( stderr, "memorizer9: malformed closure\n" );
				return( false );
			}

			V.ch[ 0 ].push_back( c );
		}

		if( d.over )
		{
			fprintf( stderr, "memorizer9: model stream exhausted\n" );
			return( false );
		}

		// the model channel must be consumed to its end, not merely parsed: trailing bytes
		// mean the declared length and the coded trie disagree
		if( l1 > d.pos + 5 )
		{
			fprintf( stderr, "memorizer9: model stream has trailing bytes\n" );
			return( false );
		}
	}

	// the rebuilt closure must be the one the encoder shipped
	{
		const long long nk = (long long) V.edge.size();
		std::uint32_t acc = 0;

		for( size_t i = 0; i < V.edge.size(); i++ )
		{
			std::uint32_t h = 2166136261u;
			h = ( h ^ (std::uint32_t) V.depth[ i ] ) * 16777619u;
			h = ( h ^ (std::uint32_t) V.edge[ i ] ) * 16777619u;
			acc += h;
		}

		if( nk != (long long) keptEnc || acc != hashEnc )
		{
			fprintf( stderr, "memorizer9: rebuild mismatch: nodes %lld vs %u\n", nk, keptEnc );
			return( false );
		}
	}

	// the decoded size is attacker-controlled: bound it in 64-bit arithmetic before
	// it can overflow or ask for an impossible allocation
	const std::uint64_t m64 = (std::uint64_t) srcLen * 2;

	if( m64 > MAX_POSITIONS )
	{
		fprintf( stderr, "memorizer9: declared size is out of range\n" );
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
		fprintf( stderr, "memorizer9: declared size does not match the coded streams\n" );
		return( false );
	}

	std::vector< unsigned char > x;

	try
	{
		x.assign( (size_t) M, 0 );
	}
	catch( const std::exception& )
	{
		fprintf( stderr, "memorizer9: cannot allocate the decoded buffer\n" );
		return( false );
	}

	if( M > 0 ) x[ 0 ] = firstNib;

	rcDec d3;
	d3.init( src.data() + o3, l3 );

	// the mirror of the encoder: the node the position lands on supplies the mode and
	// the probability that the mode is right, and on a miss one symbol among the others
	for( int t = 1; t < M; t++ )
	{
		// A crafted size can ask for far more positions than the stream codes. The
		// reader flags exhaustion the moment it happens, so check periodically rather
		// than only after the loop: the work then stops within a few thousand
		// positions of the stream running dry, instead of running to whatever the
		// header happened to declare.
		if( ( t & 4095 ) == 0 && d3.over )
		{
			fprintf( stderr, "memorizer9: coded stream exhausted\n" );
			return( false );
		}

		const size_t node = (size_t) V.walk( x, t );
		const int m = V.mode( (int) node );
		bool hit = false;

		if( m < 0 )
		{
			// nothing is known here yet, so there is no bit to read: the symbol
			// arrives on its own, exactly as the encoder sent it
			x[ (size_t) t ] = (unsigned char) d3.tree( &V.res[ node * 16 ], 4 );
		}
		else
		{
			hit = ( d3.bitPD( (std::uint16_t) V.rate[ node ].prob() ) == 0 );

			if( hit )
			{
				x[ (size_t) t ] = (unsigned char) m;
			}
			else
			{
				const int rk = d3.tree( &V.res[ node * 16 ], 4 );

				// the inverse of the encoder: ascending order, one test on the value
				// produced so far. A rank the encoder never produces would name a
				// symbol that is not a nibble, so refuse it rather than emit a byte
				// the input could not have contained.
				int tr = rk;

				if( tr >= m ) { tr++; }

				if( tr > 15 )
				{
					fprintf( stderr, "memorizer9: residue rank out of range\n" );
					return( false );
				}

				x[ (size_t) t ] = (unsigned char) tr;
			}
		}

		// the same update the encoder made, in the same order
		const int got = (int) x[ (size_t) t ];

		V.observe( (int) node, got, m >= 0, hit );
	}

	if( d3.over )
	{
		fprintf( stderr, "memorizer9: coded stream exhausted\n" );
		return( false );
	}

	// An under-declared size decodes fewer positions than the stream encodes, which
	// leaves the section reader short of its declared length. The only slack a correct
	// stream leaves is the coder's five flush bytes, so anything beyond that is a
	// stream whose declared size does not match what it codes.
	if( l3 > d3.pos + 5 )
	{
		fprintf( stderr, "memorizer9: declared size does not match the coded streams\n" );
		return( false );
	}

	try
	{
		dst.assign( (size_t) srcLen, 0 );
	}
	catch( const std::exception& )
	{
		fprintf( stderr, "memorizer9: cannot allocate the decoded buffer\n" );
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
		fprintf( stderr, "usage: memorizer9 c|d|t <in> [out]  |  memorizer9 t <in> [S]\n" );
		return( 2 );
	}

	const std::string modeStr = argv[ 1 ];

	if( modeStr != "c" && modeStr != "d" && modeStr != "t" )
	{
		fprintf( stderr, "usage: memorizer9 c|d|t <in> [out]  |  memorizer9 t <in> [S]\n" );
		return( 2 );
	}

	const char mode = modeStr[ 0 ];

	if( argc < 4 && mode != 't' )
	{
		fprintf( stderr, "usage: memorizer9 c|d|t <in> [out]  |  memorizer9 t <in> [S]\n" );
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

	// One support threshold for every input, swept over the whole corpus and reported in
	// the README: the total falls from 700,511 bytes at 6 to a minimum of 691,707 at 20,
	// and rises again past it. The surface is shallow rather than flat, so the value is a
	// measured minimum and not a sharp one.
	params P;

	// `t` takes an optional support threshold so the pruning can be swept without
	// changing the format: memorizer9 t <in> [S]
	if( mode == 't' && argc >= 4 )
	{
		const int sv = atoi( argv[ 3 ] );

		if( sv < 1 || sv > 255 )
		{
			fprintf( stderr, "memorizer9: support threshold must be 1..255\n" );
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
		catch( const std::exception& )
		{
			fprintf( stderr, "memorizer9: cannot allocate the model\n" );
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
			fprintf( stderr, "memorizer9: CHANNEL SUM MISMATCH\n" );
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
		catch( const std::exception& )
		{
			fprintf( stderr, "memorizer9: cannot allocate the model\n" );
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
		catch( const std::exception& )
		{
			fprintf( stderr, "memorizer9: cannot allocate the model\n" );
			return( 2 );
		}

		const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::vector< unsigned char > back;
		bool okDec = false;

		try
		{
			okDec = decode( C.out, back );
		}
		catch( const std::exception& )
		{
			fprintf( stderr, "memorizer9: cannot allocate the model\n" );
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
		fprintf( stderr, "memorizer9: CHANNEL SUM MISMATCH\n" );
		return( 2 );
	}

	return( same ? 0 : 1 );
}
