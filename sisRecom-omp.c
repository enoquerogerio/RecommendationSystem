#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ALEATORIO ((double)random() / (double) RAND_MAX)
#define MAX_ELEMENTS 100000

typedef struct {
    int row;
    int col;
    double value;
} MatrixElement;


void preenche_aleatorio_LR(int nU, int nI, int nF, double L[nU][nF], double R[nF][nI] ) {
	srandom(0);
	int i, j;
	for(i = 0; i < nU; i++)
		for(j = 0; j < nF; j++)
			L[i][j] = ALEATORIO / (double) nF;
	for(i = 0; i < nF; i++)
		for(j = 0; j < nI; j++)
			R[i][j] = ALEATORIO / (double) nF;
}


void multiplyMatrices(int rows1, int cols1,
                      int cols2, double matrix1[rows1][cols1], double matrix2[cols1][cols2], double result[rows1][cols2]) {
    int i, j, k;

    // Multiplicacao das matrizes
    for (i = 0; i < rows1; i++) {
        for (j = 0; j < cols2; j++) {
            result[i][j] = 0;
            for (k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

double calcular_somatorio(double expressao, int tamanho) {
    double soma = 0.0;
    for (int i = 0; i < tamanho; i++) {
        soma += expressao;
    }
    return soma;
}

int main(int argc, char *argv[]) {

    clock_t start, end;
    double cpu_time_used;

    // Marca o tempo de início
    start = clock();
    
    if (argc != 2) {
        printf("Uso: %s <arquivo de entrada>\n", argv[0]);
        return 1;
    }

    FILE *arquivo = fopen(argv[1], "r");
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo %s.\n", argv[1]);
        return 1;
    }
	int nF, nU, nI;
    int iterations;
    double alpha;
    int latent_features;
    int rows, cols, non_zero_elements_count;

    // Le as informacoes do arquivo
    fscanf(arquivo, "%d", &iterations);
    fscanf(arquivo, "%lf", &alpha);
    fscanf(arquivo, "%d", &latent_features);
    fscanf(arquivo, "%d %d %d", &rows, &cols, &non_zero_elements_count);

    // Le a matriz de elementos
    MatrixElement elements[MAX_ELEMENTS];
    for (int i = 0; i < non_zero_elements_count; i++) {
        fscanf(arquivo, "%d %d %lf", &elements[i].row, &elements[i].col, &elements[i].value);
    }

    // Fecha o arquivo
    fclose(arquivo);

	nF = latent_features;
	nU = rows;
	nI = cols;

    // Preenche o restante da matriz com zeros
    double matrix[rows][cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0.0;
        }
    }

    // Preenche a matriz com os elementos fornecidos no arquivo
    for (int i = 0; i < non_zero_elements_count; i++) {
        matrix[elements[i].row][elements[i].col] = elements[i].value;
    }

//    printf("Matriz:\n");
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            printf("%.1lf ", matrix[i][j]);
//        }
//        printf("\n");
//    }
    

    double L[nU][nF];
    double R[nF][nI];
    double Laux[nU][nF];
    double Raux[nF][nI];
    double B[nU][nI];
    
    preenche_aleatorio_LR(nU, nI, nF, L, R);
    multiplyMatrices(nU, nF, nI, L, R, B);
    //updateMatrices(nU, nI, nF, alpha, L, R, matrix);

    //copia das matrizes
    for(int i = 0; i < nU; i++){
        for(int j = 0; j < nF; j++){
            Laux[i][j] = L[i][j];
        }
    }

    for(int i = 0; i < nF; i++){
        for(int j = 0; j < nI; j++){
            Raux[i][j] = R[i][j];
        }
    }

    //atualizacao dos valores de L e R

//atualizacao dos valores de L e R
for(int loop = 0; loop < iterations; loop++) {
    // Atualização de L
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < nU; i++) {
        for(int k = 0; k < nF; k++) {
            double sum_lr = 0.0;
            #pragma omp simd reduction(+:sum_lr)
            for(int j = 0; j < nI; j++) {
                if(matrix[i][j] != 0.0) {
                    sum_lr += 2 * (matrix[i][j] - B[i][j]) * (-Raux[k][j]); // Usando Raux para calcular
                }
            }
            L[i][k] = Laux[i][k] - alpha * sum_lr;
        }
    }

    // Atualização de R
    #pragma omp parallel for collapse(2)
    for(int k = 0; k < nF; k++) {
        for(int j = 0; j < nI; j++) {
            double sum_rl = 0.0;
            #pragma omp simd reduction(+:sum_rl)
            for(int i = 0; i < nU; i++) {
                if(matrix[i][j] != 0.0) {
                    sum_rl += 2 * (matrix[i][j] - B[i][j]) * (-Laux[i][k]); // Usando Laux para calcular
                }
            }
            R[k][j] = Raux[k][j] - alpha * sum_rl;
        }
    }

    // Atualiza as cópias de L e R para a próxima iteração
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < nU; i++) {
        for(int k = 0; k < nF; k++) {
            Laux[i][k] = L[i][k];
        }
    }

    #pragma omp parallel for collapse(2)
    for(int k = 0; k < nF; k++) {
        for(int j = 0; j < nI; j++) {
            Raux[k][j] = R[k][j];
        }
    }

    // Recalcula a matriz B para a próxima iteração
    multiplyMatrices(nU, nF, nI, L, R, B);
}

  //Exibe a matriz resultante B
//    printf("Matriz resultante B:\n");
//    for (int i = 0; i < nU; i++) {
//        for (int j = 0; j < nI; j++) {
//            printf("%.2lf ", B[i][j]);
//        }
//       printf("\n");
//    }

    // Saida do programa
    double major; 
	int recomendedItem;
    printf("\n\n");
    for (int i = 0; i < nU; i++) {
    	major = 0;
        for (int j = 0; j < nI; j++) {
        	if ((B[i][j] > major) && (matrix[i][j] == 0.0)) {
        		major = B[i][j];
        		recomendedItem = j;
			}
        }
        printf("%d ", recomendedItem);
        printf("\n");
    }

     // Marca o tempo de fim
    end = clock();

    // Calcula o tempo decorrido em segundos
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\n\nTempo de execução: %f segundos\n", cpu_time_used);

    return 0;
}
