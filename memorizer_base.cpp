/**
 * @file memorizer_base.cpp
 *
 * @version 1.1
 *
 * @brief Baseline memorizer trie model build and recall implementation.
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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <chrono>

static double ent( const std::vector< int >& h, std::size_t tot )
{
	if( tot == 0 ) return( 0.0 );

	double hm = 1.0 / (double) tot;
	double H = 0.0;

	for( std::size_t i = 0; i < h.size(); i++ )
	{
		if( h[ i ] == 0 ) continue;

		double p = (double) h[ i ] * hm;
		H -= p * std::log2( p );
	}

	return( H );
}

template< typename TK, int W, int Q >
class memorizer
{
public:
	static const TK Wmask = ( (TK) 1 << W ) - 1;
	static const TK Qmask = ( W * Q >= sizeof( TK ) * 8 ) ?
		~(TK) 0 : ( (TK) 1 << ( W * Q )) - 1;

	memorizer()
	{
		static_assert( Q > 1, "Q must be greater than 1" );
		static_assert( W * Q <= sizeof( TK ) * 8, "W*Q must fit TK bits" );

		TK m = Qmask;

		for( int d = Q - 1; d >= 0; d-- )
		{
			Qmasks[ d ] = m;
			m >>= W;
		}
	}

	void addContext( const TK ctx, const int target )
	{
		for( int d = 0; d < Q; d++ )
		{
			TK k = ctx & Qmasks[ d ];

			node& nd = trie[ d ][ k ];
			nd.support++;

			if( nd.votes == 0 || nd.mode == target )
			{
				nd.mode = target;
				nd.votes += 2;
			}
			else
			{
				nd.votes--;
			}
		}
	}

	void prune( const int S )
	{
		// Collapse and prune childs by occurrence support,
		// and reconstruct trie.

		std::unordered_map< TK, node > nt[ Q ];

		for( int d = Q - 1; d >= 1; d-- )
		{
			TK d1m = Qmasks[ d - 1 ];

			for( typename std::unordered_map< TK, node >::iterator it =
				trie[ d ].begin(); it != trie[ d ].end(); ++it )
			{
				typename std::unordered_map< TK, node >::iterator ai =
					trie[ d - 1 ].find( it->first & d1m );

				if( ai != trie[ d - 1 ].end() )
				{
					if( ai->second.mode == it->second.mode )
					{
						// Collapse if parent node has the same mode.
						continue;
					}
				}

				// Context occurrence support-based pruning.

				if( it->second.support >= S )
				{
					it->second.childs = 0;
					nt[ d ][ it->first ] = it->second;
				}
			}
		}

		for( typename std::unordered_map< TK, node >::iterator it =
			trie[ 0 ].begin(); it != trie[ 0 ].end(); ++it )
		{
			it->second.childs = 0;
		}

		for( int d = 1; d < Q; d++ )
		{
			trie[ d ].swap( nt[ d ]);
		}

		// Count per-node childs.

		for( int d = Q - 1; d >= 1; d-- )
		{
			TK d1m = Qmasks[ d - 1 ];

			for( typename std::unordered_map< TK, node >::iterator it =
				trie[ d ].begin(); it != trie[ d ].end(); ++it )
			{
				typename std::unordered_map< TK, node >::iterator ai =
					trie[ d - 1 ].find( it->first & d1m );

				if( ai != trie[ d - 1 ].end() )
				{
					ai->second.childs++;
				}
			}
		}

		trie_size = 0;

		for( int d = 0; d < Q; d++ )
		{
			trie_size += trie[ d ].size();
		}
	}

	int alphabet() const
	{
		return( (int) trie[ 0 ].size() );
	}

	std::size_t size() const
	{
		return( trie_size );
	}

	int predict( const TK ctx ) const
	{
		for( int d = Q - 1; d >= 0; d-- )
		{
			typename std::unordered_map< TK, node >::const_iterator it =
				trie[ d ].find( ctx & Qmasks[ d ]);

			if( it != trie[ d ].end() ) return( it->second.mode );
		}

		return( -1 ); // Never reached in a valid model.
	}

	// Approximate Huffman size of a trie in bits (call after prune only):
	// nodes*(H_edge + H_answer + H_structure)
	double huffSize() const
	{
		if( trie_size == 0 ) return( 0.0 );

		std::vector< int > he( 1 << W, 0 ), ha( 1 << W, 0 ),
			hc(( 1 << W ) + 1, 0 );

		for( int d = 0; d < Q; d++ )
		{
			for( typename std::unordered_map< TK, node >::const_iterator it =
				trie[ d ].begin(); it != trie[ d ].end(); ++it )
			{
				he[ it->first >> ( d * W )]++;
				ha[ it->second.mode ]++;
				hc[ it->second.childs ]++;
			}
		}

		double He = ent( he, trie_size ), Ha = ent( ha, trie_size ),
			Hs = ent( hc, trie_size );

		return (double) trie_size * ( He + Ha + Hs );
	}

private:
	struct node
	{
		int mode = 0; // Modal symbol (Boyer-Moore candidate, inexact).
		int votes = 0; // Mode's majority vote counter.

		union
		{
			int support = 0; // The number of occurrences of this context.
			int childs; // The number of child nodes after prune().
		};
	};

	std::unordered_map< TK, node > trie[ Q ]; // Trie partitioned by ctx length.
	TK Qmasks[ Q ];
	std::size_t trie_size = 0;
};

// Walks an input stream and forms context keys.
template< typename TK, int W, int Q >
class ctx_walker
{
public:
	explicit ctx_walker( const std::vector< char >& raw )
		: sp( raw.data() )
		, se( raw.data() + raw.size() )
	{
		static_assert( Q > 1, "Q must be greater than 1" );
		static_assert( W * Q <= sizeof( TK ) * 8, "W*Q must fit TK bits" );

		advance();
	}

	int advance()
	{
		while( true )
		{
			if( cbits >= W )
			{
				TK sym = carry & memorizer< TK, W, Q >::Wmask;
				ctx = ctx << W | sym;
				carry >>= W;
				ctx &= memorizer< TK, W, Q >::Qmask;
				cbits -= W;

				return( (int) sym );
			}

			do
			{
				if( sp >= se )
				{
					if( cbits >= W )
					{
						break;
					}

					return( -1 );
				}

				carry |= (TK) (unsigned char) *sp << cbits;
				cbits += 8;
				sp++;
			} while( cbits <= (int) ( sizeof( carry ) - 1 ) * 8 );
		}
	}

	operator TK() const
	{
		return( ctx );
	}

private:
	const char* sp;
	const char* se;
	TK ctx = 0;
	TK carry = 0;
	int cbits = 0;
};

template< typename TK, int W, int Q >
void evalModel( const std::vector< char >& raw, const int S )
{
	const std::chrono::steady_clock::time_point t0 =
		std::chrono::steady_clock::now();

	memorizer< TK, W, Q > M1;
	{
		ctx_walker< TK, W, Q > cw( raw );

		while( true )
		{
			TK ctx = cw;
			int next_sym = cw.advance();

			if( next_sym < 0 ) break;

			M1.addContext( ctx, next_sym );
		}
	}

	M1.prune( S );

	const std::chrono::steady_clock::time_point t1 =
		std::chrono::steady_clock::now();

	ctx_walker< TK, W, Q > cw( raw );
	std::vector< int > resid( 1 << W, 0 );
	int miss = 0;
	int tot = 0;

	while( true )
	{
		int pred = M1.predict( cw );
		int next_sym = cw.advance();

		if( next_sym < 0 ) break;

		tot++;

		if( pred != next_sym )
		{
			miss++;
			resid[ next_sym ]++;
		}
	}

	const std::chrono::steady_clock::time_point t2 =
		std::chrono::steady_clock::now();

	double r1 = (double) ( tot - miss ) / (double) ( tot < 1 ? 1 : tot );
	double Hb1 = ( r1 <= 0.0 || r1 >= 1.0 ) ? 0.0 :
		( -r1 * std::log2( r1 ) - ( 1.0 - r1 ) * std::log2( 1.0 - r1 ) );

	double Hres = ent( resid, miss );
	double bt = std::chrono::duration< double >( t1 - t0 ).count();
	bt = ( bt <= 0.0 ? 1.0 : bt );
	double pt = std::chrono::duration< double >( t2 - t1 ).count();
	pt = ( pt <= 0.0 ? 1.0 : pt );

	printf( "  W=%d Q=%d S=%d A=%d | nodes %zu | recall %.2f%% | "
		"H(hit) %.2f | H(resid) %.2f\n"
		"    model size ~%.0f B\n    build %.2f MB/s | predict %.2f MB/s\n",
		W, Q, S, M1.alphabet(), M1.size(), 100.0 * r1, Hb1, Hres,
		M1.huffSize() / 8.0, 1e-6 * (double) raw.size() / bt,
		1e-6 * (double) raw.size() / pt );
}

int main( int argc, char** argv )
{
	if( argc < 2 )
	{
		fprintf( stderr, "file name not specified\n" );
		return( 1 );
	}

	const long long fsize_max = 500000000;
	const char* const fname = argv[ 1 ];

	std::vector< char > raw;
	{
		std::ifstream in( fname, std::ios::binary );

		if( !in.is_open() )
		{
			fprintf( stderr, "cannot open %s\n", fname );
			return( 2 );
		}

		in.seekg( 0, std::ios::end );
		long long fsize = in.tellg();
		in.seekg( 0, std::ios::beg );

		if( fsize <= 0 || fsize > fsize_max )
		{
			fprintf( stderr, "Zero-length or too large file %lld\n", fsize );
			return( 3 );
		}

		raw.assign(( std::istreambuf_iterator< char >( in )),
			std::istreambuf_iterator< char >() );
	}

	printf( "\n%s | %zu bytes\n", fname, raw.size() );

	evalModel< std::uint32_t, 4, 8 >( raw, 20 );
	evalModel< std::uint32_t, 2, 8 >( raw, 20 );
	evalModel< std::uint32_t, 8, 4 >( raw, 5 );

	return( 0 );
}
