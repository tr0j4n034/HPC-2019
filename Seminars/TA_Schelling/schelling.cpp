// schelling.cpp

/*******************************************************************************
 
    PARALLEL SCHELLING MODEL CELLULAR AUTOMATON

********************************************************************************

  * Compile with: 'mpicxx -std=c++11 schelling.cpp -o schelling.out'
  * Run with: 'mpirun -n <proc> ./schelling.out \
        <iter> <size> <thresh> <frac> <empty>'

*******************************************************************************/

#include "schelling.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <random>
#include <unistd.h>
#include <inttypes.h>

////////////////////////////////////////////////////////////////////////////////
//  Constants
////////////////////////////////////////////////////////////////////////////////
static const uint8_t SHUFFLE_DEPTH = 3;
const uint8_t City::_DUMP_HEADER_LEN = 52;
const char * City::_DUMP_HEADER
    = "P2\n#%05d %.5lf %.5lf %.5lf\n%05d %05d\n%05d  ";

////////////////////////////////////////////////////////////////////////////////
//  Happy tree friends :)
////////////////////////////////////////////////////////////////////////////////
// print human-readable discrepancy
static const char * ZZZ = "000";
static const char * PNZ = "+-0";
static const char * PPN = "++-";
static const char * NNP = "--+";

static const char * PrintDiscrepancy(discrepancy_t discrepancy)
{
    if (discrepancy == DISCR_PNZ) { return PNZ; }
    else if (discrepancy == DISCR_PPN) { return PPN; }
    else if (discrepancy == DISCR_NNP) { return NNP; }
    else { return ZZZ; }
}

// min
inline const uint_t & min(const uint_t & first, const uint_t & second)
{
    return (first < second)? first: second;
}

// nearest to zero
inline const int & absmin(
    const uint_t sign, const int & first, const int & second
)
{
    if (sign) { return (first < second)? first: second; }
    else { return (first < second)? second: first; }
}

// swap
template<typename T>
inline void Swap(T & first, T & second)
{
    T tmp = second;
    second = first;
    first = tmp;

    return;
}

// Fisher-Yates shuffle
template<typename T>
void Shuffle(randgen_t & generator, const uint_t size, T * arr)
{
    udist_t distribution(0, size - 1);

    for (int d = 0; d < SHUFFLE_DEPTH; ++d)
    {
        for (int s = 0; s < size; ++s)
        {
            Swap<T>(arr[s], arr[distribution(generator)]);
        }
    }

    return;
}

// Fisher-Yates shuffle, double addressation
void Shuffle(
    randgen_t & generator, const uint_t size, const uint_t * inds, uint8_t * arr
)
{
    udist_t distribution(0, size - 1);

    for (int d = 0; d < SHUFFLE_DEPTH; ++d)
    {
        for (int s = 0; s < size; ++s)
        {
            Swap(arr[inds[s]], arr[inds[distribution(generator)]]);
        }
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Get/Set methods
////////////////////////////////////////////////////////////////////////////////
inline uint_t City::GetFullHeight(void) const
{
    return _size[0] + _border[0] + _border[1];
}

inline uint_t City::GetFullIndex(const int row, const int col) const
{
    return (row + _border[0]) * _size[1] + col;
}

// not used in current version
inline uint8_t City::GetVicinitySize(const int row, const int col) const
{ 
    uint8_t r = (row > 0 || _border[0]) + (row < _size[0] - 1 || _border[1]);
    uint8_t c = (col > 0) + (col < _size[1] - 1);
    uint8_t tot = r + c;

    return tot? (1 << (tot - 1)) + ((tot >> 1) & (r && c)): 0;
}

inline uint8_t * City::GetFirstRow(void)
{
    return _houses + (_border[0]? _size[1]: 0);
}

inline uint8_t * City::GetLastRow(void)
{
    return _houses + (_size[0] - !(_border[0])) * _size[1];
}

inline uint8_t * City::GetUpperGhost(void)
{
    return _houses;
}

inline uint8_t * City::GetLowerGhost(void)
{
    return _houses + (_size[0] + _border[0]) * _size[1];
}

inline uint8_t & City::GetHouse(const uint_t ind) const
{
    return _houses[(_border[0]? _size[1]: 0) + ind];
}

inline uint8_t & City::GetHouse(const int row, const int col) const
{
    return _houses[(row + _border[0]) * _size[1] + col];
}

inline void City::SetHouse(const uint_t ind, const uint8_t house)
{
    _houses[(_border[0]? _size[1]: 0) + ind] = house;

    return;
}

inline void City::SetHouse(const int row, const int col, const uint8_t house)
{
    _houses[(row + _border[0]) * _size[1] + col] = house;

    return;
}

// not counting in total
inline void City::AssessHouse(const uint_t ind, uint_t * weights)
{
    ++(weights[GetHouse(ind)]);

    return;
}

// also counting in total
template<typename T>
inline void City::AssessHouse(const int row, const int col, T * weights)
{
    ++(weights[GetHouse(row, col)]);
    ++(weights[3]);

    return;
}

void City::GetState(uint_t * weights)
{
    memset(weights, 0, 3 * sizeof(uint_t));

    for (int s = 0; s < _size[0] * _size[1]; ++s) { AssessHouse(s, weights); }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Structors
////////////////////////////////////////////////////////////////////////////////
City::City(void):
    _prank(0),
    _psize(0),
    _size{0, 0},
    _border{0, 0},
    _weights{0, 0, 0},
    _houses(NULL),
    _thresh(0),
    _frac(0),
    _empty(0),
    _locstate{0, 0},
    _totstate{0, 0},
    _moving(NULL),
    _partstate(NULL),
    _partrank(NULL),
    _offset(0),
    _buffer(NULL)
{}

City::City(
    const uint_t size, const double thresh, const double frac,
    const double empty
):
    _offset(0),
    _size{0, size},
    _thresh(thresh),
    _frac(frac),
    _empty(empty),
    _border{0, 0},
    _weights{0, 0, 0},
    _locstate{0, 0},
    _totstate{0, 0},
    _partstate(NULL),
    _partrank(NULL),
    _buffer(NULL)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &_prank);
    MPI_Comm_size(MPI_COMM_WORLD, &_psize);

    // set height of the map
    _size[0] = _size[1] / _psize + (_prank < _size[1] % _psize);

    // set offset for output
    if (_prank)
    {
        _offset
            = _DUMP_HEADER_LEN + (
                _prank * (_size[1] / _psize) + min(_prank, _size[1] % _psize)
            ) * 2 * _size[1];
    }

    // set borders if process has any data
    if (_size[0])
    {
        _border[0] = (_prank > 0);

        // in case this is the last process with data
        if (_prank < _size[1] - 1)
        {
            _border[1] = (_prank < _psize - 1);
            _buffer = (char *)malloc(2 * _size[1]);
        }
    }

    _houses = (uint8_t *)malloc(GetFullHeight() * _size[1]);
    _moving = (uint_t *)malloc(_size[0] * _size[1] * sizeof(uint_t));

    _generator.seed(int(MPI_Wtime() * 10000) ^ _prank);
    ddist_t distribution{_frac, _empty, 1. - _frac - _empty};

    // set houses randomly
    for (int s = 0; s < _size[0] * _size[1]; ++s)
    {
        SetHouse(s, distribution(_generator));
    }
    
    // shuffle houses
    Shuffle(_generator, _size[0] * _size[1], GetFirstRow()); 

    // aggregated info (only on root)
    if (_psize > 1 && !_prank)
    {
        _partrank = (uint_t *)malloc(_psize * sizeof(uint_t));
        _partstate = (uint_t *)malloc(3 * _psize * sizeof(uint_t));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

City::~City(void)
{
    if (_houses) { free(_houses); }
    if (_moving) { free(_moving); }
    if (_partrank) { free(_partrank); }
    if (_partstate) { free(_partstate); }
    if (_buffer) { free(_buffer); }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Redistribution methods
////////////////////////////////////////////////////////////////////////////////
// Send/Recv ghosts between adjacent processes
void City::ExchangeGhosts(void)
{
    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);

    if (_border[0])
    {
        MPI_Isend(
            GetFirstRow(), _size[1], MPI_UINT8_T, _prank - 1, _prank,
            MPI_COMM_WORLD, &request
        );

        MPI_Irecv(
            GetUpperGhost(), _size[1], MPI_UINT8_T, _prank - 1, _prank - 1,
            MPI_COMM_WORLD, &request
        );
    }

    if (_border[1])
    {
        MPI_Isend(
            GetLastRow(), _size[1], MPI_UINT8_T, _prank + 1, _prank,
            MPI_COMM_WORLD, &request
        );

        MPI_Irecv(
            GetLowerGhost(), _size[1], MPI_UINT8_T, _prank + 1, _prank + 1,
            MPI_COMM_WORLD, &request
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

// decise the necessity of relocation
inline int City::Decise(
    const int row, const int col, const uint8_t * vicstate
) const
{
    return GetHouse(row, col) == 1
        || vicstate[2 - GetHouse(row, col)] > _thresh * vicstate[3];
}

// determine local state
void City::FindMoving(void)
{
    uint8_t vicstate[4];

    // reset local state
    memset(_locstate, 0, 4 * sizeof(uint_t));

    for (int row = 0; row < _size[0]; ++row)
    {
        for (int col = 0; col < _size[1]; ++col)
        {
            // assess vicinity state for next house
            memset(vicstate, 0, 4);

            if (row || _border[0])
            {
                AssessHouse(row - 1, col, vicstate);
                if (col) { AssessHouse(row - 1, col - 1, vicstate); }

                if (col < _size[1] - 1)
                {
                    AssessHouse(row - 1, col + 1, vicstate);
                }
            }

            if (row < _size[0] - 1 || _border[1])
            {
                AssessHouse(row + 1, col, vicstate);
                if (col) { AssessHouse(row + 1, col - 1, vicstate); }

                if (col < _size[1] - 1)
                {
                    AssessHouse(row + 1, col + 1, vicstate);
                }
            }

            if (col) { AssessHouse(row, col - 1, vicstate); }
            if (col < _size[1] - 1) { AssessHouse(row, col + 1, vicstate); }

            if (Decise(row, col, vicstate))
            {
                _moving[_locstate[3]] = GetFullIndex(row, col);
                AssessHouse(row, col, _locstate);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

// generate random state with precalculated probability
// using multinomial distribution
void City::GuessState(void)
{
    ddist_t distribution{
        double(_totstate[0]), double(_totstate[1]), double(_totstate[2])
    };

    memset(_locstate, 0, 3 * sizeof(uint_t));

    for (int u = 0; u < _locstate[3]; ++u)
    {
        ++(_locstate[distribution(_generator)]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

// detect discrepancy
discrepancy_t City::DetectDiscrepancy(const int * weights, uint8_t * inds)
{
    discrepancy_t discrepancy = DISCR_ZZZ;

    inds[0] = 0;
    inds[1] = 1;
    inds[2] = 2;

    // (0)??
    if (!(weights[inds[0]]))
    {
        Swap(inds[0], inds[2]);

        // (-)?0
        if (weights[inds[0]] < 0)
        {
            Swap(inds[0], inds[1]);
            discrepancy = DISCR_PNZ;
        }
        // (+)?0
        else if (weights[inds[0]] > 0) { discrepancy = DISCR_PNZ; }
    }
    // (-)??
    else if (weights[inds[0]] < 0)
    {
        Swap(inds[0], inds[1]);

        // (0)-?
        if (!(weights[inds[0]]))
        {
            Swap(inds[0], inds[2]);
            discrepancy = DISCR_PNZ;
        }
        // (-)-?
        else if (weights[inds[0]] < 0) { discrepancy = DISCR_NNP; }
        // (+)-?
        else
        {
            // +-(0)
            if (!(weights[inds[2]])) { discrepancy = DISCR_PNZ; }
            // +-(-)
            else if (weights[inds[2]] < 0)
            {
                Swap(inds[0], inds[2]);
                discrepancy = DISCR_NNP;
            }
            // +-(+)
            else {
                Swap(inds[1], inds[2]);
                discrepancy = DISCR_PPN;
            }
        }
    }
    // (+)??
    else
    {
        // +(0)?
        if (!(weights[inds[1]]))
        {
            Swap(inds[1], inds[2]);
            discrepancy = DISCR_PNZ;
        }
        // +(-)?
        else if (weights[inds[1]] < 0)
        {
            // +-(0)
            if (!(weights[inds[2]])) { discrepancy = DISCR_PNZ; }
            // +-(-)
            else if (weights[inds[2]] < 0)
            {
                Swap(inds[0], inds[2]);
                discrepancy = DISCR_NNP;
            }
            // +-(+)
            else {
                Swap(inds[1], inds[2]);
                discrepancy = DISCR_PPN;
            }
        }
        // +(+)?
        else { discrepancy = DISCR_PPN; }
    }

    return discrepancy;
}

// randomly equilibrate states
void City::Equilibrate(void)
{
    int state[3];

    _partrank[0] = 0;

    // state = _partsize
    state[0] = _partstate[0];
    state[1] = _partstate[1];
    state[2] = _partstate[2];

    // initialize ranks, sum-reduce sizes
    for (int p = 1; p < _psize; ++p)
    {
        _partrank[p] = p;
        state[0] += _partstate[3 * p];
        state[1] += _partstate[3 * p + 1];
        state[2] += _partstate[3 * p + 2];
    }

    // shuffle ranks
    Shuffle(_generator, _psize, _partrank);

    state[0] -= _totstate[0];
    state[1] -= _totstate[1];
    state[2] -= _totstate[2];

    // detect not matching components of precalculated total state  
    uint8_t inds[3];
    discrepancy_t discrepancy = DetectDiscrepancy(state, inds);

    // 000 -- nothing to do
    if (discrepancy == DISCR_ZZZ) { return; }
    // +-0 / ++- / --+
    else
    {
        int sign = (discrepancy != DISCR_NNP);

        int first; // random of 0 or 1
        int second = 2;
        ddist_t pos{1, 1}; // 0 and 1 with equal probability

        uint_t ch; // random of 0, 1 or 2
        udist_t chunk(1, 3);

        uint_t p; // random
        udist_t proc(0, _psize - 1);

        if (discrepancy == DISCR_PNZ)
        {
            pos = ddist_t{1};
            second = 1;
        };

        if (discrepancy == DISCR_NNP) { chunk = udist_t(-3, -1); };

        // repeat while there is discrepancy
        while (state[inds[0]] || state[inds[1]])
        {
            // update discrepancy specifiers
            if (!(state[inds[0]]) || !(state[inds[1]]))
            {
                discrepancy = DISCR_PNZ;
                sign = 1; 
                second = 1;
                pos = ddist_t{1};
                chunk = udist_t(1, 3);

                // (0)??
                if (!(state[inds[0]])) { Swap(inds[0], inds[2]); }
                // ?(0)?
                else { Swap(inds[1], inds[2]); }

                // (-)?0
                if (state[inds[0]] < 0) { Swap(inds[0], inds[1]); }
            }

            // set random process
            p = proc(_generator);
            // set weight to cut
            first = pos(_generator);

            // skip processes till it is possible to nonzero cut
            while (sign && !(_partstate[3 * _partrank[p] + inds[first]]))
            {
                ++p; 
                if (p == _psize) { p = 0; }
            }

            // determine chunk
            ch = sign?
                min(
                    min(chunk(_generator), state[inds[first]]),
                    _partstate[3 * _partrank[p] + inds[first]]
                ):
                absmin(sign, chunk(_generator), state[inds[first]]);

            // skip processes till it is possible to nonzero cut
            while (!sign && _partstate[3 * _partrank[p] + inds[second]] < -ch)
            {
                ++p;
                if (p == _psize) { p = 0; }
            }

            // equilibrate states
            state[inds[first]] -= ch;
            _partstate[3 * _partrank[p] + inds[first]] -= ch;

            state[inds[second]] += ch;
            _partstate[3 * _partrank[p] + inds[second]] += ch;
        }
    }

    return;
}

// actual redistibution of weiths by houses
void City::Redistribute(void)
{
    if (_psize > 1)
    {
        // reduce process states
        MPI_Allreduce(
            _locstate, _totstate, 3, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD
        );

        // predict local states with possible discrepancy
        GuessState();

        // gather process states
        MPI_Gather(
            _locstate, 3, MPI_UNSIGNED, _partstate, 3, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );

        // redistribute to processes
        if (!_prank) { Equilibrate(); }

        // scatter redistributed weights
        MPI_Scatter(
            _partstate, 3, MPI_UNSIGNED, _locstate, 3, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );
    }

    int u = 0;

    // night watch
    for ( ; u < _locstate[0]; ++u) { _houses[_moving[u]] = 0; }
    // wasteland
    for ( ; u < _locstate[0] + _locstate[1]; ++u) { _houses[_moving[u]] = 1; }
    // white walkers
    for ( ; u < _locstate[3]; ++u) { _houses[_moving[u]] = 2; }

    // shuffle moving houses
    Shuffle(_generator, _locstate[3], _moving, _houses);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Result dump methods
////////////////////////////////////////////////////////////////////////////////
void City::FileDump(const uint_t iteration)
{
    uint8_t tmp;
    char filename[32];
    snprintf(filename, 32 * sizeof(char), "dump_%d_%d.ppm", _prank, iteration);

    FILE * file = fopen(filename, "wb");

    fprintf(file, "P2\n#\n");
    fprintf(file, "%05d %05d\n%05d", _size[1], _size[0], 2);

    for (int row = 0; row < _size[0]; ++row)
    {
        fprintf(file, "\n");
        for (int col = 0; col < _size[1]; ++col)
        {
            tmp = GetHouse(row, col);

            fprintf(file, "%d ", tmp);
        }
    }

    fprintf(file, "\n");

    fclose(file);

    return;
}

void City::ParallelFileDump(const uint_t iteration)
{
    MPI_Barrier(MPI_COMM_WORLD);

    uint8_t tmp;
    snprintf(_buffer, 64, "dump_%d.ppm", iteration);

    MPI_Status status;
    MPI_Offset offset = _offset * sizeof(char);
    
    MPI_File file;
    MPI_File_open(
        MPI_COMM_WORLD, _buffer, MPI_MODE_WRONLY | MPI_MODE_CREATE,
        MPI_INFO_NULL, &file
    );
    MPI_File_set_view(
        file, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL
    );

    if (!_prank)
    {
        snprintf(
            _buffer, _DUMP_HEADER_LEN, _DUMP_HEADER,
            iteration, _thresh, _frac, _empty, _size[1], _size[1], 2
        );
        MPI_File_write(file, _buffer, _DUMP_HEADER_LEN, MPI_CHAR, &status);
    }

    for (int row = 0; row < _size[0]; ++row)
    {
        _buffer[0] = '\n';
        _buffer[1] = GetHouse(row, 0) + '0';

        for (int col = 1; col < _size[1]; ++col)
        {
            _buffer[col << 1] = ' ';
            _buffer[(col << 1) + 1] = GetHouse(row, col) + '0';
        }

        MPI_File_write(file, _buffer, 2 * _size[1], MPI_CHAR, &status);
    }

    if (_prank == _psize - 1)
    {
        _buffer[0] = '\n';
        MPI_File_write(file, _buffer, 1, MPI_CHAR, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);
    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Iteration process 
////////////////////////////////////////////////////////////////////////////////
void City::Iterate(const uint_t iterations)
{
    CheckState(0);
    ParallelFileDump(0);

    for (int i = 1; i <= iterations; ++i)
    {
        ExchangeGhosts();
        FindMoving();
        Redistribute();

        CheckState(i);
        ParallelFileDump(i);
    }

    return;
}

// check state
void City::CheckState(const uint_t iteration)
{
    uint_t totweights[3];

    GetState(_weights);

    MPI_Reduce(
        _weights, totweights, 3, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD
    );

    if (!_prank)
    {
        printf(
            "%d WEIGHTS: %d %d %d\n",
            iteration, totweights[0], totweights[1], totweights[2]
        );

        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 5)
    {
        int prank;
        MPI_Comm_rank(MPI_COMM_WORLD, &prank);

        if (!prank)
        {
            fprintf(
                stderr, "Not enough arguments: "
                "iterations, size, threshold, "
                "first fraction probability, empty probability\n"
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 42);
    }

    uint_t iter;
    uint_t size;
    double thresh;
    double frac;
    double empty;

    sscanf(argv[1], "%u", &iter); 
    sscanf(argv[2], "%u", &size); 
    sscanf(argv[3], "%lf", &thresh); 
    sscanf(argv[4], "%lf", &frac); 
    sscanf(argv[5], "%lf", &empty); 

    City city(size, thresh, frac, empty);
    city.Iterate(iter);

    MPI_Finalize();

    return 0;
}

// schelling.cpp
