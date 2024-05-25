#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ALEATORIO ((double)random() / (double) RAND_MAX)

typedef struct {
    int row;
    int col;
    double value;
} MatrixElement;

void preenche_aleatorio_LR(int nU, int nI, int nF, double L[nU][nF], double R[nF][nI]) {
    srandom(0);
    int i, j;
    for (i = 0; i < nU; i++)
        for (j = 0; j < nF; j++)
            L[i][j] = ALEATORIO / (double) nF;
    for (i = 0; i < nF; i++)
        for (j = 0; j < nI; j++)
            R[i][j] = ALEATORIO / (double) nF;
}

void multiplyMatrices(int rows1, int cols1, int cols2, double matrix1[rows1][cols1], double matrix2[cols1][cols2], double result[rows1][cols2]) {
    int i, j, k;
    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Uso: %s <arquivo de entrada>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    FILE *arquivo;
    int nF, nU, nI;
    int iterations;
    double alpha;
    int latent_features;
    int rows, cols, non_zero_elements_count;

    MatrixElement *elements = NULL;

    if (rank == 0) {
        arquivo = fopen(argv[1], "r");
        if (arquivo == NULL) {
            printf("Erro ao abrir o arquivo %s.\n", argv[1]);
            MPI_Finalize();
            return 1;
        }

        fscanf(arquivo, "%d", &iterations);
        fscanf(arquivo, "%lf", &alpha);
        fscanf(arquivo, "%d", &latent_features);
        fscanf(arquivo, "%d %d %d", &rows, &cols, &non_zero_elements_count);

        elements = (MatrixElement *)malloc(non_zero_elements_count * sizeof(MatrixElement));
        for (int i = 0; i < non_zero_elements_count; i++) {
            fscanf(arquivo, "%d %d %lf", &elements[i].row, &elements[i].col, &elements[i].value);
        }

        fclose(arquivo);
    }

    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&latent_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&non_zero_elements_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        elements = (MatrixElement *)malloc(non_zero_elements_count * sizeof(MatrixElement));
    }

    MPI_Bcast(elements, non_zero_elements_count * sizeof(MatrixElement), MPI_BYTE, 0, MPI_COMM_WORLD);

    nF = latent_features;
    nU = rows;
    nI = cols;

    double L[nU][nF];
    double R[nF][nI];
    double Laux[nU][nF];
    double Raux[nF][nI];
    double B[nU][nI];
    double matrix[nU][nI];

    for (int i = 0; i < nU; i++) {
        for (int j = 0; j < nI; j++) {
            matrix[i][j] = 0.0;
        }
    }

    for (int i = 0; i < non_zero_elements_count; i++) {
        matrix[elements[i].row][elements[i].col] = elements[i].value;
    }

    preenche_aleatorio_LR(nU, nI, nF, L, R);

    if (rank == 0) {
        multiplyMatrices(nU, nF, nI, L, R, B);
    }

    for (int loop = 0; loop < iterations; loop++) {
        MPI_Bcast(L, nU * nF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(R, nF * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < nU; i++) {
                for (int j = 0; j < nF; j++) {
                    Laux[i][j] = L[i][j];
                }
            }
            for (int i = 0; i < nF; i++) {
                for (int j = 0; j < nI; j++) {
                    Raux[i][j] = R[i][j];
                }
            }
        }

        MPI_Bcast(Laux, nU * nF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(Raux, nF * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int rows_per_process = nU / size;
        int start_row = rank * rows_per_process;
        int end_row = (rank + 1) * rows_per_process;
        if (rank == size - 1) {
            end_row = nU;
        }

        for (int i = start_row; i < end_row; i++) {
            for (int k = 0; k < nF; k++) {
                double sum_lr = 0.0;
                for (int j = 0; j < nI; j++) {
                    if (matrix[i][j] != 0.0) {
                        sum_lr += 2 * (matrix[i][j] - B[i][j]) * (-Raux[k][j]);
                    }
                }
                L[i][k] = Laux[i][k] - alpha * sum_lr;
            }
        }

        for (int k = 0; k < nF; k++) {
            for (int j = 0; j < nI; j++) {
                double sum_rl = 0.0;
                for (int i = start_row; i < end_row; i++) {
                    if (matrix[i][j] != 0.0) {
                        sum_rl += 2 * (matrix[i][j] - B[i][j]) * (-Laux[i][k]);
                    }
                }
                MPI_Allreduce(MPI_IN_PLACE, &sum_rl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                R[k][j] = Raux[k][j] - alpha * sum_rl;
            }
        }

        MPI_Gather(&L[start_row][0], rows_per_process * nF, MPI_DOUBLE, &L[0][0], rows_per_process * nF, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            multiplyMatrices(nU, nF, nI, L, R, B);
        }

        MPI_Bcast(B, nU * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double major;
        int recommendedItem;
        for (int i = 0; i < nU; i++) {
            major = 0;
            for (int j = 0; j < nI; j++) {
                if ((B[i][j] > major) && (matrix[i][j] == 0.0)) {
                    major = B[i][j];
                    recommendedItem = j;
                }
            }
            printf("%d\n", recommendedItem);
        }
    }

    free(elements);
    MPI_Finalize();
    return 0;
}