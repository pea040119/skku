all:
	g++ -g -O2 -o seqsort sequential_sort.cpp
	g++ -g -o textgen text_gen.cpp

sort:
	nvcc -o debug/sort_$(type) src/main.cpp src/sort.cu
	debug/sort_$(type) input.txt 100000 0 10 256 $(type) > output/sort_$(type).txt

