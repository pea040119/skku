all:
	g++ -fopenmp -o hnsw main.cpp hnsw.cpp -lm
test:
	g++ -fopenmp -o hnsw main.cpp hnsw.cpp -lm
	./hnsw -k100 -t40 -w 1 > test.txt