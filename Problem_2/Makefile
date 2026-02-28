in=GRCh38_latest_rna.fna

all: Exp1 Exp2

# Experiment 1 Compilations
Exp1: compute_average_TF_Exp1_critical compute_average_TF_Exp1_atomic compute_average_TF_Exp1_locks

compute_average_TF_Exp1_critical: compute_average_TF_Exp1_critical.c
	gcc -fopenmp -O3 -Wall -o compute_average_TF_Exp1_critical compute_average_TF_Exp1_critical.c -lm -std=c99

compute_average_TF_Exp1_atomic: compute_average_TF_Exp1_atomic.c
	gcc -fopenmp -O3 -Wall -o compute_average_TF_Exp1_atomic compute_average_TF_Exp1_atomic.c -lm -std=c99

compute_average_TF_Exp1_locks: compute_average_TF_Exp1_locks.c
	gcc -fopenmp -O3 -Wall -o compute_average_TF_Exp1_locks compute_average_TF_Exp1_locks.c -lm -std=c99

# Experiment 2 Compilation
Exp2: compute_average_TF_Exp2_schedule

compute_average_TF_Exp2_schedule: compute_average_TF_Exp2_schedule.c
	gcc -fopenmp -O3 -Wall -o compute_average_TF_Exp2_schedule compute_average_TF_Exp2_schedule.c -lm -std=c99

clean:
	$(RM) compute_average_TF_Exp1_critical compute_average_TF_Exp1_atomic compute_average_TF_Exp1_locks compute_average_TF_Exp2_schedule *.csv

# Run Commands
run_critical:
	./compute_average_TF_Exp1_critical $(in) OUTPUT_crit.csv TIME_crit.csv 8

run_atomic:
	./compute_average_TF_Exp1_atomic $(in) OUTPUT_atom.csv TIME_atom.csv 8

run_locks:
	./compute_average_TF_Exp1_locks $(in) OUTPUT_locks.csv TIME_locks.csv 8

run_exp2:
	export OMP_SCHEDULE="static" && ./compute_average_TF_Exp2_schedule $(in) OUT_static.csv TIME_static.csv 8
	export OMP_SCHEDULE="dynamic" && ./compute_average_TF_Exp2_schedule $(in) OUT_dyn.csv TIME_dyn.csv 8
	export OMP_SCHEDULE="guided" && ./compute_average_TF_Exp2_schedule $(in) OUT_guid.csv TIME_guid.csv 8