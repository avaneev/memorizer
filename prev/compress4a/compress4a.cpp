/**
 * @file compress4a.cpp
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
// compress4a -- a memoriser corpus compressor whose model is built from one
// radix sort of the packed context keys.
//
//   compress4a c <in> <out>    compress
//   compress4a d <in> <out>    decompress
//   compress4a t <in>          sizes, channels, ratio and a round-trip check
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
//   Support pruning: a node costs about 8.7 bits to transmit while a repaired miss
//   costs about 2.9, so a context seen fewer than S times is not worth shipping.
//
// KEPT versus CLOSURE. These are different sets, and the distinction is the whole
// construction:
//   kept(n)    = n survives the collapse and its support is at least S;
//   closure(n) = kept(n), or n lies on a path from the root to a kept node.
// A closure node is pure structure: it carries no answer of its own, only the
// "inherit" code, and exists so the decoder can reach a kept descendant.
//
// FIVE CHANNELS, charged separately, summing to the file size:
//   C1 model1    the closure of the memoriser, DFS pre-order
//   C2 model2    the closure of the corrector (0 while the corrector is off)
//   C3 flag1     hit or miss, once per position
//   C4 flag2     corrected or not, once per MISS (0 while the corrector is off)
//   C5 residual  the true symbol where the model misses
//
// CONSTRUCTION. No pointer trie and no hash table build the model: one radix sort
// of the packed Q-symbol keys orders the positions, and every shorter depth is read
// off that same order by shifting the key, because positions sharing a (d)-suffix
// are a contiguous run of the (d+1)-ordered array. Counts, modal answers and
// supports of every context of every length come from one sort and Q linear scans;
// the collapse's parent test is a binary search over the next-shorter depth's
// already-sorted group keys. The closure is then emitted in one DFS pass.
//
// COMPILE
//   g++ -O2 -std=c++11 -pedantic-errors -Wall -Wextra -o compress4a.exe compress4a.cpp
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

struct params
{
	int Q = 8;                 // memoriser context length in symbols
	int S = 6;                 // support threshold (12 for small inputs)
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

	void tree( std::uint16_t* p, int bits, int v )
	{
		int m = 1;

		for( int i = bits - 1; i >= 0; i-- )
		{
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

	void reset()
	{
		edge.assign( 1, 0 );
		ans.assign( 1, -1 );
		ch.assign( 1, std::vector< int >() );
		depth.assign( 1, 0 );
	}

	int addNode( int parent, int e )
	{
		edge.push_back( (unsigned char) e );
		ans.push_back( -1 );
		ch.push_back( std::vector< int >() );
		depth.push_back( depth[ (size_t) parent ] + 1 );
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

	// DFS pre-order: edge | answer code (0..15 answer, 16 inherit) | child count
	void ser( rcEnc& e, std::uint16_t* pe, std::uint16_t* pa, std::uint16_t* pc,
		int node ) const
	{
		e.tree( pe, 4, (int) edge[ (size_t) node ] );
		e.tree( pa, 5, ( ans[ (size_t) node ] >= 0 ) ? (int) ans[ (size_t) node ] : 16 );
		e.tree( pc, 5, (int) ch[ (size_t) node ].size() );

		for( size_t c = 0; c < ch[ (size_t) node ].size(); c++ )
		{
			ser( e, pe, pa, pc, ch[ (size_t) node ][ c ] );
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
};

static int parseNode( rcDec& d, closureView& V, std::uint16_t* pe, std::uint16_t* pa,
	std::uint16_t* pc, int parentDepth, int Q, long long maxNodes )
{
	if( parentDepth + 1 > Q || (long long) V.edge.size() >= maxNodes )
	{
		return( -1 );
	}

	const int edge = d.tree( pe, 4 );
	const int code = d.tree( pa, 5 );
	const int nc = d.tree( pc, 5 );

	if( code < 0 || code > 16 || nc < 0 || nc > 16 )
	{
		return( -1 );
	}

	const int id = (int) V.edge.size();
	V.edge.push_back( (unsigned char) edge );
	V.ans.push_back( ( code == 16 ) ? -1 : code );
	V.ch.push_back( std::vector< int >() );
	V.depth.push_back( parentDepth + 1 );

	for( int i = 0; i < nc; i++ )
	{
		const int c = parseNode( d, V, pe, pa, pc, parentDepth + 1, Q, maxNodes );

		if( c < 0 )
		{
			return( -1 );
		}

		V.ch[ (size_t) id ].push_back( c );
	}

	return( id );
}

// descend by the context symbol, keeping the deepest answer seen
template< class TR >
static inline int walk( const TR& V, const std::vector< unsigned char >& x, int t, int Q,
	int rootMode )
{
	int node = 0;
	int last = rootMode;

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
		}
	}

	return( last );
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

static std::vector< unsigned char > readFile( const char* path, bool& ok )
{
	std::ifstream in( path, std::ios::binary );
	ok = !!in;

	if( !ok )
	{
		return( std::vector< unsigned char >() );
	}

	std::string s( ( std::istreambuf_iterator< char >( in ) ),
		std::istreambuf_iterator< char >() );
	return( std::vector< unsigned char >( s.begin(), s.end() ) );
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

// header: magic 4 | srcLen 4 | Q, firstNib, rootMode, rootNch 4 | kept 4 | hash 4 |
// three section lengths 12 = 32 bytes
static const size_t HEADER = 32;

// the largest input either side will handle, in symbols: 2^30 nibbles = 512 MiB of
// bytes. One constant for both directions, so the encoder can never produce a stream
// the decoder refuses.
static const std::uint64_t MAX_POSITIONS = (std::uint64_t) 1 << 30;

// ---------------------------------------------------------------------------
struct codec
{
	std::vector< unsigned char > out;
	long long c1 = 0, c3 = 0, c5 = 0;
};

static bool encode( const std::vector< unsigned char >& src, const params& P, codec& C )
{
	if( (std::uint64_t) src.size() * 2 > MAX_POSITIONS )
	{
		fprintf( stderr, "compress4a: input too large\n" );
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

	// the memoriser pass: hit or miss, and the true symbol at every miss
	std::vector< unsigned char > resid;

	for( int t = 1; t < M; t++ )
	{
		const int p = walk( mo.cl, x, t, P.Q, mo.rootMode );

		if( p != (int) x[ (size_t) t ] )
		{
			// the true symbol is never the prediction, so code its RANK among the
			// other fifteen: rank = true with the predicted value removed
			const int tr = (int) x[ (size_t) t ];
			resid.push_back( (unsigned char) ( ( tr < p ) ? tr : tr - 1 ) );
		}
	}

	std::vector< unsigned char > m1, f1, rs;
	{
		rcEnc e;
		std::uint16_t pe[ 16 ], pa[ 32 ], pc[ 32 ];

		for( int i = 0; i < 16; i++ ) pe[ i ] = 1024;
		for( int i = 0; i < 32; i++ ) pa[ i ] = pc[ i ] = 1024;

		for( size_t c = 0; c < mo.cl.ch[ 0 ].size(); c++ )
		{
			mo.cl.ser( e, pe, pa, pc, mo.cl.ch[ 0 ][ c ] );
		}

		e.flush();
		m1 = e.out;
	}
	{
		rcEnc e;
		std::uint16_t pr = 1024;

		for( int t = 1; t < M; t++ )
		{
			const int p = walk( mo.cl, x, t, P.Q, mo.rootMode );
			e.bit( pr, ( p == (int) x[ (size_t) t ] ) ? 1 : 0 );
		}

		e.flush();
		f1 = e.out;
	}
	{
		rcEnc e;
		std::uint16_t pr[ 16 ];

		for( int i = 0; i < 16; i++ ) pr[ i ] = 1024;

		for( size_t i = 0; i < resid.size(); i++ )
		{
			e.tree( pr, 4, (int) resid[ i ] );
		}

		e.flush();
		rs = e.out;
	}

	C.c1 = (long long) m1.size();
	C.c3 = (long long) f1.size();
	C.c5 = (long long) rs.size();

	C.out.clear();
	C.out.push_back( 'C' );
	C.out.push_back( '3' );
	C.out.push_back( 'c' );
	C.out.push_back( '1' );
	putU32( C.out, (std::uint32_t) src.size() );
	C.out.push_back( (unsigned char) P.Q );
	C.out.push_back( (unsigned char) ( ( M > 0 ) ? x[ 0 ] : 0 ) );
	C.out.push_back( (unsigned char) mo.rootMode );
	C.out.push_back( (unsigned char) mo.cl.ch[ 0 ].size() );
	putU32( C.out, (std::uint32_t) mo.kept );
	putU32( C.out, mo.cl.keptHash() );
	putU32( C.out, (std::uint32_t) m1.size() );
	putU32( C.out, (std::uint32_t) f1.size() );
	putU32( C.out, (std::uint32_t) rs.size() );
	// checksum 1: the header fields, appended before the payload so a decoder can
	// verify the header without interpreting any of it
	putU32( C.out, fnv( C.out.data(), C.out.size() ) );

	const size_t payloadStart = C.out.size();

	C.out.insert( C.out.end(), m1.begin(), m1.end() );
	C.out.insert( C.out.end(), f1.begin(), f1.end() );
	C.out.insert( C.out.end(), rs.begin(), rs.end() );

	// checksum 2: the payload only, so the two halves are diagnosed separately
	putU32( C.out, fnv( C.out.data() + payloadStart, C.out.size() - payloadStart ) );
	return( true );
}

static bool decode( const std::vector< unsigned char >& src, std::vector< unsigned char >& dst )
{
	if( src.size() < HEADER + 8 || src[ 0 ] != 'C' || src[ 1 ] != '3' || src[ 2 ] != 'c'
		|| src[ 3 ] != '1' )
	{
		fprintf( stderr, "compress4a: not a stream or truncated header\n" );
		return( false );
	}

	// checksum 1: the header, verified before any field is read
	std::uint32_t hdrStored = 0;

	for( int b = 0; b < 4; b++ )
	{
		hdrStored |= (std::uint32_t) src[ HEADER + (size_t) b ] << ( 8 * b );
	}

	if( fnv( src.data(), HEADER ) != hdrStored )
	{
		fprintf( stderr, "compress4a: header checksum mismatch\n" );
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
		fprintf( stderr, "compress4a: data checksum mismatch\n" );
		return( false );
	}

	size_t p = 4;
	const std::uint32_t srcLen = getU32( src, p );
	const int Q = src[ p++ ];
	// the codec is defined for one context length and one corrector length; a stream
	// that declares anything else is not one we produced
	if( Q != 8 )
	{
		fprintf( stderr, "compress4a: unsupported context length\n" );
		return( false );
	}

	const unsigned char firstNib = src[ p++ ];

	if( firstNib > 15 )
	{
		fprintf( stderr, "compress4a: first symbol is not a nibble\n" );
		return( false );
	}
	const unsigned char rootMode = src[ p++ ];

	if( rootMode > 15 )
	{
		fprintf( stderr, "compress4a: root answer is not a symbol\n" );
		return( false );
	}
	const int rootNch = src[ p++ ];
	const std::uint32_t keptEnc = getU32( src, p );
	const std::uint32_t hashEnc = getU32( src, p );
	const std::uint32_t l1 = getU32( src, p );
	const std::uint32_t l3 = getU32( src, p );
	const std::uint32_t l5 = getU32( src, p );

	if( (std::uint64_t) l1 + l3 + l5
		!= (std::uint64_t) ( bodyEnd - payloadStart ) )
	{
		fprintf( stderr, "compress4a: section lengths do not match the file\n" );
		return( false );
	}

	const size_t o1 = payloadStart;
	const size_t o3 = o1 + l1;
	const size_t o5 = o3 + l3;

	closureView V;
	V.edge.push_back( 0 );
	V.ans.push_back( -1 );
	V.ch.push_back( std::vector< int >() );
	V.depth.push_back( 0 );
	{
		rcDec d;
		d.init( src.data() + o1, l1 );
		std::uint16_t pe[ 16 ], pa[ 32 ], pc[ 32 ];

		for( int i = 0; i < 16; i++ ) pe[ i ] = 1024;
		for( int i = 0; i < 32; i++ ) pa[ i ] = pc[ i ] = 1024;

		for( int i = 0; i < rootNch; i++ )
		{
			const int c = parseNode( d, V, pe, pa, pc, 0, Q, 8 * (long long) l1 + 8 );

			if( c < 0 )
			{
				fprintf( stderr, "compress4a: malformed closure\n" );
				return( false );
			}

			V.ch[ 0 ].push_back( c );
		}

		if( d.over )
		{
			fprintf( stderr, "compress4a: model stream exhausted\n" );
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
			fprintf( stderr, "compress4a: rebuild mismatch: kept %lld vs %u\n", nk, keptEnc );
			return( false );
		}
	}

	// the decoded size is attacker-controlled: bound it in 64-bit arithmetic before
	// it can overflow or ask for an impossible allocation
	const std::uint64_t m64 = (std::uint64_t) srcLen * 2;

	if( m64 > MAX_POSITIONS )
	{
		fprintf( stderr, "compress4a: declared size is out of range\n" );
		return( false );
	}

	const int M = (int) m64;
	std::vector< unsigned char > x;

	try
	{
		x.assign( (size_t) M, 0 );
	}
	catch( const std::exception& e )
	{
		fprintf( stderr, "compress4a: cannot allocate the decoded buffer\n" );
		return( false );
	}

	if( M > 0 ) x[ 0 ] = firstNib;

	rcDec d3, d5;
	d3.init( src.data() + o3, l3 );
	d5.init( src.data() + o5, l5 );
	std::uint16_t p3 = 1024;
	std::uint16_t p5[ 16 ];

	for( int i = 0; i < 16; i++ ) p5[ i ] = 1024;

	for( int t = 1; t < M; t++ )
	{
		if( d3.bit( p3 ) != 0 )
		{
			x[ (size_t) t ] = (unsigned char) walk( V, x, t, Q, (int) rootMode );
		}
		else
		{
			const int rank = d5.tree( p5, 4 );

			// undo the rank: a rank at or above the prediction shifts up by one
			const int pred = walk( V, x, t, Q, (int) rootMode );
			x[ (size_t) t ] = (unsigned char) ( ( rank < pred ) ? rank : rank + 1 );
		}
	}

	if( d3.over || d5.over )
	{
		fprintf( stderr, "compress4a: coded stream exhausted\n" );
		return( false );
	}

	// an under-declared size decodes fewer positions than the streams encode, which
	// leaves the section readers short of their declared lengths
	if( l3 > d3.pos + 16 || l5 > d5.pos + 16 )
	{
		fprintf( stderr, "compress4a: declared size does not match the coded streams\n" );
		return( false );
	}

	dst.assign( (size_t) srcLen, 0 );

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
		fprintf( stderr, "usage: compress4a c|d|t <in> [out]\n" );
		return( 2 );
	}

	const std::string modeStr = argv[ 1 ];

	if( modeStr != "c" && modeStr != "d" && modeStr != "t" )
	{
		fprintf( stderr, "usage: compress4a c|d|t <in> [out]\n" );
		return( 2 );
	}

	const char mode = modeStr[ 0 ];

	if( argc < 4 && mode != 't' )
	{
		fprintf( stderr, "usage: compress4a c|d|t <in> [out]\n" );
		return( 2 );
	}

	bool ok = false;
	std::vector< unsigned char > src = readFile( argv[ 2 ], ok );

	if( !ok )
	{
		fprintf( stderr, "cannot open %s\n", argv[ 2 ] );
		return( 2 );
	}

	// A single support threshold, swept on the Canterbury corpus: S = 6 was the best
	// or within a few bytes of it on every text file, including the smallest ones, so
	// there is no small-input special case.
	params P;

	// `t` takes an optional support threshold so the pruning can be swept without
	// changing the format: compress4a t <in> [S]
	if( mode == 't' && argc >= 4 )
	{
		const int sv = atoi( argv[ 3 ] );

		if( sv >= 1 && sv <= 64 )
		{
			P.S = sv;
		}
	}

	if( mode == 'c' )
	{
		codec C;

		if( !encode( src, P, C ) )
		{
			return( 2 );
		}

		if( !writeFile( argv[ 3 ], C.out ) )
		{
			fprintf( stderr, "cannot write %s\n", argv[ 3 ] );
			return( 2 );
		}

		const long long sum = C.c1 + C.c3 + C.c5 + (long long) HEADER + 8;

		printf( "channels: C1 %lld | C2 0 | C3 %lld | C4 0 | C5 %lld | header %lld |"
			" checksum 8 | sum %lld = file %d\n", C.c1, C.c3, C.c5, (long long) HEADER,
			sum, (int) C.out.size() );

		if( sum != (long long) C.out.size() )
		{
			fprintf( stderr, "compress4a: CHANNEL SUM MISMATCH\n" );
			return( 2 );
		}

		printf( "compressed %d -> %d bytes (%.4fx)\n", (int) src.size(), (int) C.out.size(),
			(double) C.out.size() / (double) std::max< size_t >( 1, src.size() ) );
		return( 0 );
	}

	if( mode == 'd' )
	{
		std::vector< unsigned char > out;

		if( !decode( src, out ) )
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

		if( !encode( src, P, C ) )
		{
			return( 2 );
		}

		const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::vector< unsigned char > back;
		const bool okDec = decode( C.out, back );
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
	const long long sum = C.c1 + C.c3 + C.c5 + (long long) HEADER + 8;

	printf( "input %d B | compressed %d B | ratio %.4f | %.2f MB/s compress |"
		" %.2f MB/s decompress | round-trip %s\n", (int) src.size(), (int) C.out.size(),
		(double) C.out.size() / (double) std::max< size_t >( 1, src.size() ), cMBs, dMBs,
		same ? "OK" : "FAILED" );
	printf( "channels: C1 %lld | C2 0 | C3 %lld | C4 0 | C5 %lld | header %lld | checksum 8"
		" | sum %lld = file %d\n", C.c1, C.c3, C.c5, (long long) HEADER, sum,
		(int) C.out.size() );

	if( sum != (long long) C.out.size() )
	{
		fprintf( stderr, "compress4a: CHANNEL SUM MISMATCH\n" );
		return( 2 );
	}

	return( same ? 0 : 1 );
}
