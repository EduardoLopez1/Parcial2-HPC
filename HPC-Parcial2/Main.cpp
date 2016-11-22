//Library C
#include "mpi.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//Library C++
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>

void Mult_Matrix_Cuda(double *h_a, double *h_b, double *h_c, int height, int width_a, int width_b);
void Mult_Matrix_MPI(double *a, double *b, double *c, int rows, int h, int w);
void Print_Matrix(double *M, int w, int h);
bool Check_Matriz(double *M1, double *M2, int w, int h);
typedef std::chrono::duration<double> duration_seconds;
using namespace std;

#define NumberRows_A 1000
#define NumberCols_A 1000
#define NumberCols_B 1000
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main(int argc, char *argv[])
{
    int numtasks,                 /* number of tasks in partition */
        taskid,                   /* a task identifier */
        numworkers,               /* number of worker tasks */
        source,                   /* task id of message source */
        dest,                     /* task id of message destination */
        mtype, /* message type */ /* cols of matrix A sent to each worker */
        averow, extra;            /* used to determine cols sent to each worker */

    uint64_t i, rc, rows = 0, offset; /* misc */

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    if (numtasks < 2)
    {
        cerr << "Necesita como mínimo dos tareas de MPI" << endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
        return EXIT_FAILURE;
    }

    numworkers = numtasks - 1;

    /**************************** master task ************************************/
    if (taskid == MASTER)
    {
        double *Matrix_a, *Matrix_b, *Device_Matrix_c, *Mpi_Matrix_d; /* result matrix C */

        Matrix_a = (double *)malloc(NumberRows_A * NumberCols_A * sizeof(double));
        Matrix_b = (double *)malloc(NumberCols_A * NumberCols_B * sizeof(double));
        Device_Matrix_c = (double *)malloc(NumberRows_A * NumberCols_B * sizeof(double));
        Mpi_Matrix_d = (double *)malloc(NumberRows_A * NumberCols_B * sizeof(double));

        printf("mpi_mm has started with %d tasks.\n", numtasks);
        printf("Initializing arrays...\n");
        //LLenando las matrices..
        for (i = 0; i < NumberRows_A * NumberCols_A; i++)
        {
            Matrix_a[i] = (double)i;
            Matrix_b[i] = (double)i;
        }

        // for (i = 0; i < NumberCols_A * NumberCols_B; i++)
        // {
        //     Matrix_b[i] = (double)i;
        // }

        // cout << "Matriz A" << endl;
        // Print_Matrix(Matrix_a, NumberCols_A, NumberRows_A);
        // cout << "Matriz B" << endl;
        // Print_Matrix(Matrix_b, NumberCols_B, NumberCols_A);
        /* Send matrix data to the worker tasks */

        //
        averow = NumberRows_A / numworkers;
        extra = NumberRows_A % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        printf("numworkers : %d\n", numworkers);
        printf("AVEROW : %d\n", averow);
        printf("EXTRA : %d\n", extra);

        //TIME
        auto start = std::chrono::high_resolution_clock::now();

        for (dest = 1; dest <= numworkers; dest++)
        {
            rows = (dest <= extra) ? averow + 1 : averow;
            printf("Sending %ld cols to task %d offset = %ld\n", rows, dest, offset);
            //offset -> permite decir que poicion del vector empieza el bloque y de tamaño rows
            MPI_Send(&offset, 1, MPI_UINT64_T, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_UINT64_T, dest, mtype, MPI_COMM_WORLD);
            //bloques de fila ...
            MPI_Send(&Matrix_a[offset * NumberCols_A], rows * NumberCols_A, MPI_DOUBLE, dest, mtype,
                     MPI_COMM_WORLD);
            MPI_Send(Matrix_b, NumberRows_A * NumberCols_B, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elap = std::chrono::duration_cast<duration_seconds>(end - start);
        std::cout << "Total time: " << elap.count() << "s\n";

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_UINT64_T, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_UINT64_T, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&Mpi_Matrix_d[offset * NumberCols_B], rows * NumberCols_B, MPI_DOUBLE, source, mtype,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(&Device_Matrix_c[offset * NumberCols_B], rows * NumberCols_B, MPI_DOUBLE, source, mtype,
                     MPI_COMM_WORLD, &status);
            printf("Received results from task %d\n", source);
        }
        // cout << "Device" << endl;
        // Print_Matrix(Device_Matrix_c, NumberCols_B, NumberRows_A);
        // cout << "MPI" << endl;
        // Print_Matrix(Mpi_Matrix_d, NumberCols_B, NumberRows_A);

        string igual = ((Check_Matriz(Device_Matrix_c, Mpi_Matrix_d, NumberCols_B, NumberRows_A) == 1) ? "true" : "false");
        cout << "Iguales? -> " << igual << endl;
        free(Matrix_a);
        free(Matrix_b);
        free(Device_Matrix_c);
        free(Mpi_Matrix_d);
    }

    /**************************** worker task ************************************/
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_UINT64_T, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_UINT64_T, MASTER, mtype, MPI_COMM_WORLD, &status);

        double *Matrix_a, *Matrix_b, *Device_Matrix_c, *Mpi_Matrix_d;

        Matrix_a = (double *)malloc(NumberRows_A * NumberCols_A * sizeof(double));
        Matrix_b = (double *)malloc(NumberCols_A * NumberCols_B * sizeof(double));
        Device_Matrix_c = (double *)malloc(NumberRows_A * NumberCols_B * sizeof(double));
        Mpi_Matrix_d = (double *)malloc(NumberRows_A * NumberCols_B * sizeof(double));

        MPI_Recv(Matrix_a, rows * NumberCols_A, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(Matrix_b, NumberRows_A * NumberCols_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        //CUDA
        Mult_Matrix_Cuda(Matrix_a, Matrix_b, Device_Matrix_c, rows, NumberCols_A, NumberCols_B);
        //M_PI
        Mult_Matrix_MPI(Matrix_a, Matrix_b, Mpi_Matrix_d, rows, NumberCols_A, NumberCols_B);

        mtype = FROM_WORKER;

        MPI_Send(&offset, 1, MPI_UINT64_T, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_UINT64_T, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(Mpi_Matrix_d, rows * NumberCols_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(Device_Matrix_c, rows * NumberCols_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);

        free(Matrix_a);
        free(Matrix_b);
        free(Device_Matrix_c);
        free(Mpi_Matrix_d);
    }
    MPI_Finalize();
}
void Mult_Matrix_MPI(double *a, double *b, double *c, int rows, int h, int w)
{
    for (int j = 0; j < w; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < h; k++)
            {
                c[i * w + j] += a[i * h + k] * b[k * w + j];
            }
        }
    }
}
void Print_Matrix(double *M, int w, int h)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            cout << M[i * w + j] << " ";
        }
        cout << endl;
    }
}
bool Check_Matriz(double *M1, double *M2, int w, int h)
{
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            if (M1[i * w + j] != M2[i * w + j])
            {
                return false;
            }
        }
    }
    return true;
}