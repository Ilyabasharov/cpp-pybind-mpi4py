#include <mpi.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <math.h>
#include <string>

namespace py = pybind11;
using pymod = pybind11::module;

// base variables
//#define N 100000        // number of points
#define ROOT_THREAD 0  // root thread id
#define MAX_SIZE 12    // max block size

// Initial conditions
#define F(t) sin(t)
#define T (3 * M_PI)
#define X (-2.0)

// Additional variables
#define TOLERANCE 0.000001
std::string DEFAULT_PATH = "vis/";


typedef struct pairs
{
    double x;
    double v;
} pairs;

typedef struct InitCond
{
    double T_top;
    double T_bot;
    double T_left;
    double T_right;
} InitCond;

void writeResults(const char *outputFile, double *source, double dt, long long N)
{
    FILE *output = fopen(outputFile, "w");
    if (!output)
    {
        fprintf(stderr, "Error: while opening %s: %s\n", outputFile, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double time = 0;
    
    fprintf(output, "t,x\n");
    for (int i = 0; i < N; i++)
    {
        if (fprintf(output, "%.10f, %.10f\n", time, source[i]) == 0)
        {
            fprintf(stderr, "Error: while writing the output to %s: %s\n", outputFile, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        time += dt;
    }

    fclose(output);
}

void writeMatrix(const char *outputFile, double *source, int width, int height)
{
    FILE *output = fopen(outputFile, "w");
    if (!output)
    {
        fprintf(stderr, "Error: while opening %s: %s\n", outputFile, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height - 1; j++)
            fprintf(output, "%.1lf ", source[i * width + j]);
        
        fprintf(output, "%.1lf\n", source[i * width + height - 1]);
    }
    
    fclose(output);
}

class Distributed
{
public:
    Distributed(): comm_global(MPI_COMM_WORLD) {}
    ~Distributed() {}
    
    void shoot(long long N)
    {
        int rank = 0, size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        double T_start = MPI_Wtime();

        long long n_start = (N * (rank + 0)) / size,
                  n_end   = (N * (rank + 1)) / size;

        if (rank == size - 1)
            n_end = N;

        double dt = T / N, dT = T / size,
               testShot[N] = {},
               testShotReceived[N];

        // init x and v with zero values for first shot
        double x = 0.0, v = 0.0;

        for (long long n = n_start; n < n_end; n++)
        {
            x += v * dt;
            v += F(n * dt) * dt;
            testShot[n] = x;
        }

        int recv_test_size[MAX_SIZE] = {},
            recv_test_offs[MAX_SIZE] = {};

        for (int rank = 0; rank < size; rank++)
        {
            recv_test_size[rank] = N / size;
            recv_test_offs[rank] = (N * rank) / size;
        }

        recv_test_size[size - 1] = N - recv_test_offs[size - 1];

        MPI_Gatherv(testShot + n_start, n_end - n_start, MPI_DOUBLE,
                    testShotReceived, recv_test_size, recv_test_offs, MPI_DOUBLE, ROOT_THREAD, MPI_COMM_WORLD);

        pairs p = {x, v};
        pairs p_start[MAX_SIZE] = {}, p_end[MAX_SIZE] = {};

        MPI_Gather(&p, sizeof(p), MPI_CHAR, p_end, sizeof(p), MPI_CHAR, ROOT_THREAD, MPI_COMM_WORLD);

        if (rank == ROOT_THREAD)
        {
            for (int rank = 1; rank < size; rank++)
            {
                x += v * dT;
                x += p_end[rank].x;
                v += p_end[rank].v;
            }

            // correcting v at the start of the first interval
            p_start[0].x = 0;
            p_start[0].v = (X - x) / T;

            for (int rank = 1; rank < size; rank++)
            {
                // correcting the (x, v) at the end of an interval,
                // considering that we changed (x, v) at the start of it from (0, 0)
                p_start[rank].x = p_end[rank - 1].x + p_start[rank - 1].x + p_start[rank - 1].v * dT;
                p_start[rank].v = p_end[rank - 1].v + p_start[rank - 1].v;
            }
        }

        // scattering from root corrected (x, v) at the start of the intervals
        MPI_Scatter(p_start, sizeof(p), MPI_CHAR, &p, sizeof(p), MPI_CHAR, ROOT_THREAD, MPI_COMM_WORLD);

        x = p.x;
        v = p.v;

        double correctedShotResults[N] = {};
        for (long long n = n_start; n < n_end; n++)
        {
            x += v * dt;
            v += F(n * dt) * dt;
            correctedShotResults[n] = x;
        }

        double finalResults[N] = {};
        int recv_size[MAX_SIZE] = {},
            recv_offs[MAX_SIZE] = {};

        for (int rank = 0; rank < size; rank++)
        {
            recv_size[rank] = N / size;
            recv_offs[rank] = (N * rank) / size;
        }
        recv_size[size - 1] = N - recv_offs[size - 1];

        MPI_Gatherv(correctedShotResults + n_start, n_end - n_start, MPI_DOUBLE,
                    finalResults, recv_size, recv_offs, MPI_DOUBLE, ROOT_THREAD, MPI_COMM_WORLD);

        if (rank == ROOT_THREAD)
        {
            double T_elapsed = MPI_Wtime() - T_start;
            printf("Execution time is %.5f sec.\n", T_elapsed);

            writeResults((DEFAULT_PATH + "result.csv").c_str(), finalResults, dt, N);
            writeResults((DEFAULT_PATH + "test.csv").c_str(), testShotReceived, dt, N);
        }
    }
    
    void seidel(long long N, int viz)
    {
        int rank = 0, size = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        double T_start = MPI_Wtime();
        
        InitCond ic = {100, 200, 300, 400};
        int width = 1000, height = 1000;
        double *t[2], *data = NULL;

        if (rank == 0)
            data = (double*) calloc(width * height, sizeof(*data));

        int w = width, h = (height + rank) / size;
        t[0] = (double*) calloc (w * (h + 2), sizeof(double));
        t[1] = (double*) calloc (w * (h + 2), sizeof(double));

        int* displs = NULL, *counts = NULL, i, j;
        if (rank == ROOT_THREAD)
        {
            displs = (int*) calloc(size, sizeof(int));
            counts = (int*) calloc(size, sizeof(int));
            counts[0] = w * h;
            
            for (i = 1; i < size; i++)
            {
                displs[i] = displs[i - 1] + counts[i - 1];
                counts[i] = ((height + i) / size) * width;
            }

            for (i = 0; i < height; i++)
            {
                data[i * width] = ic.T_left;
                data[(i + 1) * width - 1] = ic.T_right;
            }
            
            for (j = 0; j < width; j++)
            {
                data[j] = ic.T_top;
                data[(height - 1) * width + j] = ic.T_bot;
            }

            double alpha_x, alpha_y;
            for (int i = 1; i < height - 1; i++)
            {
                alpha_x = ((double) i) / height;
                for (int j = 1; j < width - 1; j++)
                {
                    alpha_y = ((double) j) / width;
                    data[i * width + j] = 0.5 * (ic.T_bot * alpha_x + ic.T_top * (1 - alpha_x) +
                                                ic.T_right * alpha_y + ic.T_left * (1 - alpha_y));
                }
            }
        }

        if (rank == ROOT_THREAD)
        {
            MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                         t[0], w * h, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                         t[1], w * h, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                         t[0] + w, w * h, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                         t[1] + w, w * h, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        int iter = -1, curr, next, iend = h + 1;
        double diff = 0, total_diff = 0;
        MPI_Request s_req, r_req;
        
        if (rank == ROOT_THREAD || rank == size - 1)
            iend = size == 1 ? h - 1 : h;

        while (iter < N)
        {
            iter++;
            curr = iter % 2;
            next = (iter + 1) % 2;

            if (size > 1)
            {
                if (rank == ROOT_THREAD)
                {
                    MPI_Send(t[curr] + w * (h - 1), w, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
                    MPI_Recv(t[curr] + w * h, w, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else if (rank != size - 1)
                {
                    MPI_Recv(t[curr], w, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(t[curr] + w, w, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);

                    MPI_Send(t[curr] + w * h, w, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
                    MPI_Recv(t[curr] + w * (h + 1), w, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                else
                {
                    MPI_Recv(t[curr], w, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(t[curr] + w, w, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
                }
            }

            diff = 0;
            for (i = 1; i < iend; i++)
                for (j = 1; j < w - 1; j++)
                    diff += fabs(t[0][i * w + j] - t[1][i * w + j]);

            MPI_Reduce(&diff, &total_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Bcast(&total_diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            total_diff /= height * width;

            if (total_diff < TOLERANCE && iter > 0)
                break;

            for (j = 1; j < w - 1; j++)
            {
                if (rank != ROOT_THREAD)
                {
                    MPI_Irecv(t[next] + j, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &r_req);
                    MPI_Wait(&r_req, MPI_STATUS_IGNORE);
                }
                for (i = 1; i < iend; i++)
                    t[next][i * w + j] = 0.25 * (t[next][(i - 1) * w + j] + t[curr][(i + 1) * w + j] +
                                                 t[next][i * w + (j - 1)] + t[curr][i * w + (j + 1)]);
                if (rank != size - 1)
                    MPI_Isend(t[next] + w * (iend - 1) + j, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &s_req);
            }
            
            if (viz > 0)
            {
                if (rank == ROOT_THREAD)
                    MPI_Gatherv(t[iter % 2], w * h, MPI_DOUBLE, data, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                else
                    MPI_Gatherv(t[iter % 2] + w, w * h, MPI_DOUBLE, data, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                
                if (rank == ROOT_THREAD)
                {
                    std::string filename = DEFAULT_PATH + "result_matrix_" + std::to_string(iter) + ".txt";
                    writeMatrix(filename.c_str(), data, width, height);
                }
            }
        }

        if (rank == ROOT_THREAD)
        {
            printf("Total iterations are %d, total diff: %.7lf\n", iter, total_diff);
            MPI_Gatherv(t[iter % 2], w * h, MPI_DOUBLE, data, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else
            MPI_Gatherv(t[iter % 2] + w, w * h, MPI_DOUBLE, data, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        free(t[0]);
        free(t[1]);

        if (rank == ROOT_THREAD)
        {
            double T_elapsed = MPI_Wtime() - T_start;
            printf("Execution time is %.5f sec.\n", T_elapsed);
            
            writeMatrix((DEFAULT_PATH + "result_matrix.txt").c_str(), data, width, height);
            free(data);
            free(counts);
            free(displs);
        }
    }
    
private:
  MPI_Comm comm_global;
};

PYBIND11_MODULE(mpi_lib, mmod)
{
  constexpr auto MODULE_DESCRIPTION = "Distribution math calculus lib";
  mmod.doc() = MODULE_DESCRIPTION;
  
  py::class_<Distributed>(mmod, "Distributed")    
    .def(py::init<>())
    .def("shoot", &Distributed::shoot, "Shooting method")
    .def("seidel", &Distributed::seidel, "Seidel method");
}
        