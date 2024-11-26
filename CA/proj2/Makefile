NUMS_1 := 1 2 3 4 6 9 10
NUMS_2 := 5 7 8 11
.PHONY: all test_1 test_2 run example compare print

default: run

run:
	@for num in $(NUMS_1); do \
		python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin 100 > output/output_$${num}.txt; \
	done
	@for num in $(NUMS_2); do \
		python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin testcase/proj2_$${num}_data.bin 100 > output/output_$${num}.txt; \
	done

test_1:
	python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin 100 > output/output_$${num}.txt


test_2:
	python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin testcase/proj2_$${num}_data.bin 100 > output/output_$${num}.txt

example:
	@for num in $(NUMS_1); do \
		testcase/riscv-sim testcase/proj2_$${num}_inst.bin 100 > example/example_$${num}.txt; \
	done
	@for num in $(NUMS_2); do \
		testcase/riscv-sim testcase/proj2_$${num}_inst.bin testcase/proj2_$${num}_data.bin 100 > example/example_$${num}.txt; \
	done

print:
	@for num in $(NUMS_1); do \
		python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin disassemble > disassemble/disassemble_$${num}.txt; \
	done
	@for num in $(NUMS_2); do \
		python3 src/riscv-sim.py testcase/proj2_$${num}_inst.bin disassemble > disassemble/disassemble_$${num}.txt; \
	done


compare:
	@for num in $(NUMS_1); do \
		echo "Comparing output for test case $$num"; \
		diff output/output_$${num}.txt example/example_$${num}.txt; \
	done
	@for num in $(NUMS_2); do \
		echo "Comparing output for test case $$num"; \
		diff output/output_$${num}.txt example/example_$${num}.txt; \
	done
