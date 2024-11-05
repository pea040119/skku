NUMS := 1 2 3 4 5 6 7 8

default: run

run:
	@for num in $(NUMS); do \
		python3 src/riscv-sim.py testcase/proj1_$${num}.bin > output/output_$${num}.txt; \
	done


compare:
	@if [ -z "$(NUM)" ]; then \
		echo "Please provide NUM as a make variable (e.g., make compare NUM=1)"; \
		exit 1; \
	fi
	@diff ref_output/ref_output_$(NUM).txt output/output_$(NUM).txt && echo "Output matches for NUM=$(NUM)" || echo "Output differs for NUM=$(NUM)"