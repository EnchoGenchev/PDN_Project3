#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1000000
#define GENE_ARRAY_SIZE 164000 
#define NUM_TETRANUCS   256
#define GENE_SIZE       10000

struct Genes {
    unsigned char* gene_sequences;
    int* gene_sizes;
    int num_genes;
};

struct Genes read_genes(FILE* inputFile) {
    struct Genes genes;
    genes.gene_sequences = (unsigned char*)malloc((GENE_ARRAY_SIZE * GENE_SIZE) * sizeof(unsigned char));
    genes.gene_sizes = (int*)malloc(GENE_ARRAY_SIZE * sizeof(int));
    genes.num_genes = 0;
    char line[MAX_LINE_LENGTH] = { 0 };
    fgets(line, MAX_LINE_LENGTH, inputFile);
    int currentGeneIndex = 0;
    while (fgets(line, MAX_LINE_LENGTH, inputFile)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] != '>') {
            int line_len = strlen(line);
            for (int i = 0; i < line_len; ++i) {
                char c = line[i];
                if (c == 'A' || c == 'C' || c == 'G' || c == 'T') {
                    genes.gene_sequences[genes.num_genes * GENE_SIZE + currentGeneIndex] = c;
                    currentGeneIndex += 1;
                }
            }
        } else {
            genes.gene_sizes[genes.num_genes] = currentGeneIndex;
            genes.num_genes += 1;
            currentGeneIndex = 0;
        }
    }
    genes.gene_sizes[genes.num_genes] = currentGeneIndex;
    genes.num_genes += 1;
    return genes;
}

void process_tetranucs(struct Genes genes, int* gene_TF, int gene_index) {
    int N = genes.gene_sizes[gene_index];
    unsigned char* seq = &genes.gene_sequences[gene_index * GENE_SIZE];
    if (N < 4) return;
    for (int i = 0; i <= N - 4; ++i) {
        int idx = 0;
        for (int j = 0; j < 4; ++j) {
            char c = seq[i + j];
            int val = (c == 'A') ? 0 : (c == 'C') ? 1 : (c == 'G') ? 2 : 3;
            idx = (idx << 2) | val;
        }
        gene_TF[idx]++;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) exit(-1);
    int num_threads = atoi(argv[4]);
    FILE* inputFile = fopen(argv[1], "r");
    FILE* outputFile = fopen(argv[2], "w");
    FILE* timeFile = fopen(argv[3], "w");
    
    struct Genes genes = read_genes(inputFile);
    long long* TF = (long long*)calloc(NUM_TETRANUCS, sizeof(long long));
    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();
    #pragma omp parallel default(none) shared(genes, TF)
    {
        #pragma omp for
        for (int gene_index = 0; gene_index < genes.num_genes; ++gene_index) {
            int gene_TF[NUM_TETRANUCS] = {0};
            process_tetranucs(genes, gene_TF, gene_index);
            
            #pragma omp critical
            {
                for (int t = 0; t < NUM_TETRANUCS; ++t)
                    TF[t] += gene_TF[t];
            }
        }
    }

    double end = omp_get_wtime();
    for (int i = 0; i < NUM_TETRANUCS; ++i) 
        fprintf(outputFile, "%f\n", (double)TF[i] / genes.num_genes);
    fprintf(timeFile, "%f\n", end - start);
    return 0;
}