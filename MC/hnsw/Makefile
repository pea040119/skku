.PHONY: all test

all:
	g++ -fopenmp -o output/hnsw src/main.cpp src/hnsw.cpp -lm
test:
	g++ -fopenmp -o debug/test_$(num) src/main.cpp src/hnsw.cpp -lm
	debug/test_$(num) -k100 -t40 -w 1 > output/test_$(num).txt

single:
	g++ -fopenmp -o debug/single src/main.cpp src/hnsw_single.cpp -lm
	debug/single -k100 -t40 -w 1 > output/single.txt
