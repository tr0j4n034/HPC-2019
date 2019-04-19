#ifndef SCHELLING_H
#define SCHELLING_H

/*******************************************************************************
 
    PARALLEL SCHELLING MODEL CELLULAR AUTOMATON

********************************************************************************

  * Compile with: 'mpicxx -std=c++11 schelling.cpp -o schelling.out'
  * Run with: 'mpirun -n <proc> ./schelling.out \
        <iter> <size> <thresh> <frac> <empty>'

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <random>
#include <inttypes.h>

////////////////////////////////////////////////////////////////////////////////
//  Type definitions
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint_t;

typedef std::default_random_engine randgen_t;
typedef std::uniform_int_distribution<int> udist_t;
typedef std::discrete_distribution<int> ddist_t;

typedef enum 
{
    DISCR_ZZZ = 0, // 000
    DISCR_PNZ = 1, // +-0
    DISCR_PPN = 2, // ++-
    DISCR_NNP = 3  // --+
} discrepancy_t; 

////////////////////////////////////////////////////////////////////////////////
//  Happy tree friends :)
////////////////////////////////////////////////////////////////////////////////
// print human-readable discrepancy
static const char * PrintDiscrepancy(discrepancy_t discrepancy);

// min
inline const uint_t & min(const uint_t & first, const uint_t & second);

// nearest to zero
inline const int & absmin(
    const uint_t sign, const int & first, const int & second
);

// swap
template<typename T = uint8_t>
inline void Swap(T & first, T & second);

// Fisher-Yates shuffle
template<typename T = uint8_t>
void Shuffle(randgen_t & generator, const uint_t size, T * arr);

// Fisher-Yates shuffle, double addressation
void Shuffle(
    randgen_t & generator, const uint_t size, const uint_t * inds, uint8_t * arr
);

////////////////////////////////////////////////////////////////////////////////
//  City class
////////////////////////////////////////////////////////////////////////////////
class City
{
        //====================================================================//
        //  Fields
        //====================================================================//
        // MPI specifiers
        int _prank;
        int _psize; 

        // city map 
        uint_t _size[2]; // height, width
        uint_t _border[2]; // upper, lower
        uint_t _weights[3]; // night watch, wasteland, white walkers
        uint8_t * _houses; 

        // threshold of intolerance
        double _thresh;
        // probability of night watch
        double _frac;
        // probability of wasteland
        double _empty;

        // moving houses
        uint_t _locstate[4]; // night watch, wasteland, white walkers, total
        uint_t _totstate[3]; // night watch, wasteland, white walkers
        uint_t * _moving; // indices of moving householders

        // shuffled process ranks (only on root)
        uint_t * _partrank;
        // aggregated states for each process (only on root)
        uint_t * _partstate; // night watch, wasteland, white walkers

        // I/O handling
        static const uint8_t _DUMP_HEADER_LEN;
        static const char * _DUMP_HEADER;
        uint_t _offset; // global offset for parallel file write
        char * _buffer;

        // random generator
        randgen_t _generator;
        
        //====================================================================//
        //  Get/Set methods
        //====================================================================//
        uint_t GetFullHeight(void) const;
        uint_t GetFullIndex(const int row, const int col) const;
        uint8_t GetVicinitySize(const int row, const int col) const;
        void GetState(uint_t * weights);
        
        uint8_t * GetFirstRow(void);
        uint8_t * GetLastRow(void);
        uint8_t * GetUpperGhost(void);
        uint8_t * GetLowerGhost(void);

        uint8_t & GetHouse(const uint_t ind) const;
        uint8_t & GetHouse(const int row, const int col) const;
        void SetHouse(const uint_t ind, const uint8_t house);
        void SetHouse(const int row, const int col, const uint8_t house);

        void AssessHouse(const uint_t ind, uint_t * weights);

        template<typename T>
        void AssessHouse(const int row, const int col, T * weights);

        //====================================================================//
        //  Redistribution methods
        //====================================================================//
        template<typename T>
        friend void Swap(T & first, T & second);

        template<typename T>
        friend void Shuffle(randgen_t & generator, const uint_t size, T * arr);

        friend void Shuffle(
            randgen_t & generator, const uint_t size, const uint_t * inds,
            uint8_t * arr
        );

        void ExchangeGhosts(void);
        int Decise(
            const int row, const int col, const uint8_t * vicstate
        ) const;
        void FindMoving(void);
        void GuessState(void);
        discrepancy_t DetectDiscrepancy(const int * weights, uint8_t * inds);
        void Equilibrate(void);
        void Redistribute(void);

        //====================================================================//
        //  Result dump methods
        //====================================================================//
        void FileDump(const uint_t step);
        void ParallelFileDump(const uint_t step);

    public:

        //====================================================================//
        //  Structors
        //====================================================================//
        City(void);
        City(
            const uint_t size, const double thresh, const double prob,
            const double empty
        );
        ~City(void);

        //====================================================================//
        //  Iteration process methods
        //====================================================================//
        void CheckState(const uint_t iteration);
        void Iterate(const uint_t iterations);
};

#endif // SCHELLING_H
