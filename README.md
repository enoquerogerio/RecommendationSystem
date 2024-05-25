# RecommendationSystem

Este repositório contém três versões de um sistema de recomendação: uma versão serial, uma versão paralela utilizando OpenMP e uma versão paralela utilizando MPI.ures.


## Conteúdo

- **Versão Serial**: Implementação básica e sequencial do sistema de recomendação.

- **Versão Paralela OpenMP**: Implementação paralela utilizando OpenMP para aproveitamento de múltiplos núcleos de CPU.

- **Versão Paralela MPI**:  Implementação paralela utilizando MPI para execução em múltiplos nós de um cluster.






## Pré-requisitos

Antes de compilar e executar qualquer uma das versões, certifique-se de ter as bibliotecas necessárias instaladas no seu sistema.

### Instalação das Bibliotecas
Para instalar as bibliotecas necessárias, execute os seguintes comandos:

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install libomp-dev   # Para OpenMP
sudo apt-get install mpich        # Para MPI

```
    
## Compilação e Execução

### Versão Serial
Para compilar e executar a versão serial do programa, siga os passos abaixo:

```bash
  1. gcc -o main sisRecom.c
  2. ./main dado0.in
```
### Versão Paralela OpenMP
Para compilar e executar a versão paralela utilizando OpenMP, siga os passos abaixo:

```bash
  1. gcc -fopenmp -o omp sisRecom-omp.c
  2. ./omp dado0.in
```
### Versão Paralela MPI
Para compilar e executar a versão paralela utilizando MPI, siga os passos abaixo:

```bash
  1. mpicc -o mpi sisRecom-mpi.c
  2. mpirun -np 2 ./mpi dado0.in
```
## Testes e Verificação

O arquivo de entrada ``` dado0.in ``` é utilizado para testar o programa. Os resultados gerados pelo programa podem ser comparados com arquivos de saída ``` .out ``` fornecidos, para verificar a correção da implementação.


## Licença
Este projeto está licenciado sob a licença [MIT](https://choosealicense.com/licenses/mit/).
