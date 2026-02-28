#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1000000
#define GENE_ARRAY_SIZE 164000 
#define NUM_TETRANUCS   256
#define GENE_SIZE       10000

// Store gene-data here
struct Genes {
    unsigned char* gene_sequences; 
    int* gene_sizes;               
    int num_genes;                 
}; 

// Reads in the gene-data from a file
struct Genes read_genes(FILE* inputFile) {
    struct Genes genes;
    genes.gene_sequences = (unsigned char*)malloc((GENE_ARRAY_SIZE * GENE_SIZE) * sizeof(unsigned char));
    genes.gene_sizes = (int*)malloc(GENE_ARRAY_SIZE * sizeof(int));
    genes.num_genes = 0;

    char line[MAX_LINE_LENGTH] = { 0 };
    fgets(line, MAX_LINE_LENGTH, inputFile); // Skip first header

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

// Logic to process tetranucleotides for a single gene
void process_tetranucs(struct Genes genes, int* gene_TF, int gene_index) {
    int N = genes.gene_sizes[gene_index];
    unsigned char* sequence = &genes.gene_sequences[gene_index * GENE_SIZE];

    if (N < 4) return;

    for (int i = 0; i <= N - 4; ++i) {
        int idx = 0;
        for (int j = 0; j < 4; ++j) {
            char c = sequence[i + j];
            int val = (c == 'A') ? 0 : (c == 'C') ? 1 : (c == 'G') ? 2 : 3;
            idx = (idx << 2) | val; 
        }
        gene_TF[idx]++;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("USE LIKE THIS: ./compute_average_TF_Exp1_locks input.fna average_TF.csv time.csv num_threads\n");
        exit(-1);
    }

    int num_threads = atoi(argv[4]);
    FILE* inputFile = fopen(argv[1], "r");
    FILE* outputFile = fopen(argv[2], "w");
    FILE* timeFile = fopen(argv[3], "w");

    if (!inputFile || !outputFile || !timeFile) {
        printf("ERROR: File opening failed.\n");
        exit(-2);
    }

    struct Genes genes = read_genes(inputFile);
    long long* TF = (long long*)calloc(NUM_TETRANUCS, sizeof(long long));

    // Initialize an array of locks, one for each tetranucleotide index
    omp_lock_t locks[NUM_TETRANUCS];
    for (int i = 0; i < NUM_TETRANUCS; i++) {
        omp_init_lock(&locks[i]);
    }

    omp_set_num_threads(num_threads);
    double start = omp_get_wtime();

    #pragma omp parallel default(none) shared(genes, TF, locks)
    {
        #pragma omp for
        for (int gene_index = 0; gene_index < genes.num_genes; ++gene_index) {
            int gene_TF[NUM_TETRANUCS] = {0}; 
            
            process_tetranucs(genes, gene_TF, gene_index);

            // Update the global TF array using the lock corresponding to the index
            for (int t = 0; t < NUM_TETRANUCS; ++t) {
                if (gene_TF[t] > 0) {
                    omp_set_lock(&locks[t]);
                    TF[t] += gene_TF[t];
                    omp_unset_lock(&locks[t]);
                }
            }
        }
    }

    double end = omp_get_wtime();

    // Finalize: Calculate and print averages
    for (int i = 0; i < NUM_TETRANUCS; ++i) {
        fprintf(outputFile, "%f\n", (double)TF[i] / (double)genes.num_genes);
    }

    fprintf(timeFile, "%f\n", end - start);

    // Cleanup locks and files
    for (int i = 0; i < NUM_TETRANUCS; i++) {
        omp_destroy_lock(&locks[i]);
    }

    fclose(timeFile);
    fclose(inputFile);
    fclose(outputFile);
    free(TF);
    free(genes.gene_sequences);
    free(genes.gene_sizes);

    return 0;
}