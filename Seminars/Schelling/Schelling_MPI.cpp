#include <mpi.h>
#include <random>
#include <assert.h>
#include <sstream>
#include <iostream>

// Schelling model parallelized with MPI

// Compile with 'mpic++ Schelling_MPI.cpp -o Schelling_MPI -std=c++11'
// Run with 'mpirun -n {psize} ./Schelling_MPI {coeff} {n_iter} {N}'

// by Yuriy Gabuev


class City
{
        int M, N;

        // all houses row-wise
        int *houses;

        // coefficient from [0., 1.]
        double coeff;

        // indices of houses whose inhabitants need to move
        int *wantmove;

        // number of black and white houses who want to move
        int counts[2];

    public:
        City();

        // random initialization of City with M rows and N columns
        // adds two "ghost" rows at the begining and end
        City(const int newM, const int newN, const double newcoeff);

        ~City();
        int* getCounts() {return counts;};
        void setCounts(int c0, int c1) {counts[0]=c0; counts[1]=c1;};
        void EvaluateMove();
        void ExchangeRows(int prank, int psize, MPI_Comm communicator);
        void Shuffle();
        void FileDump(int iteration, int prank);
};

City::City()
{
    M = 0;
    N = 0;
    houses = NULL;
    coeff = 0.;
    wantmove = NULL;
    counts[0] = 0;
    counts[1] = 1;
}

City::City(const int newM, const int newN, const double newcoeff)
{
    // initialize a rectangular City with two extra rows
    // to fill in the data from neighbors
    M = newM;
    N = newN;
    coeff = newcoeff;

    houses   = (int*)malloc((M+2) * N * sizeof(int));
    wantmove = (int*)malloc((M+2) * N * sizeof(int));

    std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < M+2; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            houses[i * N + j] = distribution(generator);
            wantmove[i * N + j] = -1; // stop flag
        }
    }

    return;
}

City::~City()
{
    if (houses)
    {
        free(houses);
    }

    if (wantmove)
    {
        free(wantmove);
    }

    return;
}

void City::EvaluateMove()
{
    int n_white_neighbors;  // number of white neighbors
    int cell_color;
    int vicinity_status;    // number of neighbors of the opposite color

    counts[0] = 0;
    counts[1] = 0;

    int lind, rind; // indices of first elements to the left and to the right
    for (int i = 1; i < M+1; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            n_white_neighbors = 0;
            cell_color = houses[N * i + j];

            if (j==0)
            {
                lind = N-1;
                rind = 1;
            } else if (j == N-1) {
                lind = N-2;
                rind = 0;
            } else {
                lind = j-1;
                rind = j+1;
            }

            n_white_neighbors += houses[(i - 1) * N + lind]; // row above
            n_white_neighbors += houses[(i - 1) * N + j];
            n_white_neighbors += houses[(i - 1) * N + rind];

            n_white_neighbors += houses[i * N + lind];       // current row
            n_white_neighbors += houses[i * N + rind];

            n_white_neighbors += houses[(i + 1) * N + lind]; // row below
            n_white_neighbors += houses[(i + 1) * N + j];
            n_white_neighbors += houses[(i + 1) * N + rind];

            vicinity_status = (1 - cell_color) * n_white_neighbors + cell_color * (8 - n_white_neighbors);
            if (double(vicinity_status) >= 8 * coeff)
            {
                wantmove[counts[0]+counts[1]] = i * N + j;
                counts[cell_color]++;
            }
        }
    }
    wantmove[counts[0]+counts[1]] = -1; // stop flag
}

void City::ExchangeRows(int prank, int psize, MPI_Comm communicator) {
    MPI_Status status;
    MPI_Request request;

    int neighbor_upper = (psize + prank-1) % psize;
    int neighbor_lower = (psize + prank+1) % psize;

    MPI_Isend(houses,           N, MPI_INT, neighbor_upper, 777, communicator, &request);
    MPI_Isend(houses + (M+1)*N, N, MPI_INT, neighbor_lower, 777, communicator, &request);
    MPI_Recv( houses,           N, MPI_INT, neighbor_upper, 777, communicator, &status);
    MPI_Recv( houses + (M+1)*N, N, MPI_INT, neighbor_lower, 777, communicator, &status);
}

void City::Shuffle() {
    int N0 = counts[0];
    int N1 = counts[1];
    int N = N0+N1;

    int* mask = new int[N0+N1];
    std::mt19937 generator(std::random_device{}());

    double p0;
    for (int i=0; i<N; i++) {
        p0 = (double)N0 / (N0+N1);
        std::bernoulli_distribution distribution(1-p0);
        int bit = distribution(generator);
        mask[i] = bit;
        if (bit==0) N0--;
        else        N1--;
    }

    int ind;
    for (int i=0; i<M*N; i++) {
        ind = wantmove[i];
        if (ind==-1) break;

        houses[ind] = mask[i];
    }
}

void City::FileDump(int iteration, int prank)
{
    char buffer[50]; // The filename buffer.
    snprintf(buffer, sizeof(char) * 32, "dump_%d_proc_%d.data", iteration, prank);

    FILE *file = fopen(buffer, "wb");
    int count = M * N;

    fwrite(houses+N, sizeof(int), count, file); // write M rows, starting with the second
    fclose(file);
    return;
}


int split_one(double p0, int N_all) {
    std::mt19937 generator(std::random_device{}());
    std::binomial_distribution<int> distribution(N_all, p0);

    int N0;
    N0 = distribution(generator);
    return N0;
}

int* split_all(int *N0, int *N1, int K) {
    int* N0_new = new int[K];

    int N0_total = 0; // total number of 0s
    int N1_total = 0; // total number of 1s
    for (int i=0; i<K; i++) {
        N0_total += N0[i];
        N1_total += N1[i];
    }

    double p0;
    for (int k=0; k < K-1; k++) {
        int Nk = N0[k] + N1[k];
        p0 = (double)N0_total / (N0_total+N1_total);
        int N0k_new = split_one(p0, Nk);
        N0_new[k] = N0k_new;
        N0_total -= N0k_new;
        N1_total -= (Nk-N0k_new);
    }
    N0_new[K-1] = N0_total;
    return N0_new;
}


int main(int argc, char ** argv)
{
    double coeff;
    int n_iter;
    int N;

    std::istringstream ss1(argv[1]);
    if (!(ss1 >> coeff)) {
      std::cerr << "Invalid coefficient: " << argv[1] << '\n';
    } else if (!ss1.eof()) {
      std::cerr << "Trailing characters after number: " << argv[1] << '\n';
    }

    std::istringstream ss2(argv[2]);
    if (!(ss2 >> n_iter)) {
      std::cerr << "Invalid number of iterations: " << argv[2] << '\n';
    } else if (!ss2.eof()) {
      std::cerr << "Trailing characters after number: " << argv[2] << '\n';
    }

    std::istringstream ss3(argv[3]);
    if (!(ss3 >> N)) {
      std::cerr << "Invalid N: " << argv[3] << '\n';
    } else if (!ss3.eof()) {
      std::cerr << "Trailing characters after number: " << argv[3] << '\n';
    }

    MPI_Init(&argc, &argv);
    int prank;
    int psize;

    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Comm_size(MPI_COMM_WORLD, &psize);

    assert(N % psize == 0);
    const int M = N / psize;

    City city(M, N, coeff);

    int* counts = new int[2];
    int counts_total;
    int* N0 = new int[psize];
    int* N1 = new int[psize];
    int* N0_new = new int[psize];

    for (int iter=0; iter < n_iter; iter++)
    {
        city.FileDump(iter, prank);
        city.ExchangeRows(prank, psize, MPI_COMM_WORLD);
        city.EvaluateMove();
        counts = city.getCounts();
        counts_total = counts[0] + counts[1];

        // gather black and white counts
        MPI_Gather(&counts[0], 1, MPI_INT, N0, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&counts[1], 1, MPI_INT, N1, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (prank == 0) {
            N0_new = split_all(N0, N1, psize);
        }

        MPI_Scatter(N0_new, 1, MPI_INT, &counts[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
        counts[1] = counts_total - counts[0];
        city.setCounts(counts[0], counts[1]);
        city.Shuffle();
                MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
