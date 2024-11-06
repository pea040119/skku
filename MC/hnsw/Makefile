.PHONY: all test

all:
	g++ -fopenmp -o debug/hnsw src/main.cpp src/hnsw.cpp -lm

hnsw:
	g++ -fopenmp -o debug/hnsw_$(num) src/main.cpp src/hnsw.cpp -lm
	debug/hnsw_$(num) -k100 -t40 -w1 > output/hnsw_$(num).txt

single:
	g++ -fopenmp -o debug/single src/main.cpp src/hnsw_single.cpp -lm
	debug/single -k100 -t40 -w1 > output/single.txt

single_1:
	g++ -fopenmp -o debug/single_1 src/main.cpp src/hnsw_single.cpp -lm
	debug/single_1 -k100 -t40 -w1 > output/single_1.txt

test:
	@for i in $(shell seq 1 10); do \
		echo "Iteration $$i"; \
		debug/hnsw_1 -k100 -t40 -w1 > output/test_1_$$i.txt; \
	done

run:
	g++ -fopenmp -o debug/hnsw src/main.cpp src/hnsw.cpp -lm
	debug/hnsw -k100 -t40 -w1 > output/hnsw.txt

clean:
	rm -f output/test_*
	rm -f output/hnsw_*