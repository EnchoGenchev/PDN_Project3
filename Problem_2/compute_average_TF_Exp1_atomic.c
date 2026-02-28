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
    // Finalize last gene
    genes.gene_sizes[genes.num_genes] = currentGeneIndex;
    genes.num_genes += 1;

    return genes;
}

// Logic to process tetranucleotides for a single gene
void process_tetranucs(struct Genes genes, int* gene_TF, int gene_index) {
    //process gene array
    int N = genes.gene_sizes[gene_index];
    unsigned char* sequence = &genes.gene_sequences[gene_index * GENE_SIZE];

    //need 4 nucleotides
    for (int i = 0; i <= N - 4; ++i) {
        int idx = 0;
        int valid = 1;

        //convert substring to values 0-255
        for (int j = 0; j < 4; ++j) {
            char c = sequence[i + j];
            int val;
            if (c == 'A') 
                val = 0;
            else if (c == 'C') 
                val = 1;
            else if (c == 'G') 
                val = 2;
            else if (c == 'T') 
                val = 3;
            else {
                valid = 0; //handle other characters
                break;
            }

            //idx = Window[0]*64 + Window[1]*16 + Window[2]*4 + Window[3] 
            //updates with each iteration, not all at once
            idx = (idx * 4) + val;
        }

        if (valid) {
            gene_TF[idx]++;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("USE LIKE THIS: ./compute_average_TF_Exp1_atomic input.fna average_TF.csv time.csv num_threads\n");
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
    
    // Using long long for global count to prevent overflow before averaging
    long long* TF = (long long*)calloc(NUM_TETRANUCS, sizeof(long long));

    omp_set_num_threads(num_threads);
    double start = omp_get_wtime();

    // PARALLEL REGION
    #pragma omp parallel default(none) shared(genes, TF)
    {
        #pragma omp for
        for (int gene_index = 0; gene_index < genes.num_genes; ++gene_index) {
            int gene_TF[NUM_TETRANUCS] = {0}; // Local thread-safe storage for this gene
            
            process_tetranucs(genes, gene_TF, gene_index);

            // Synchronize using atomic for each index update
            for (int t = 0; t < NUM_TETRANUCS; ++t) {
                if (gene_TF[t] > 0) {
                    #pragma omp atomic
                    TF[t] += gene_TF[t];
                }
            }
        }
    }

    double end = omp_get_wtime();

    // Output average frequencies
    for (int i = 0; i < NUM_TETRANUCS; ++i) {
        fprintf(outputFile, "%f\n", (double)TF[i] / (double)genes.num_genes);
    }

    fprintf(timeFile, "%f\n", end - start);

    // Cleanup
    fclose(timeFile);
    fclose(inputFile);
    fclose(outputFile);
    free(TF);
    free(genes.gene_sequences);
    free(genes.gene_sizes);

    return 0;
}