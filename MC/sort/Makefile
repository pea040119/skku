all:
	g++ -g -O2 -o seqsort sequential_sort.cpp
	g++ -g -o textgen text_gen.cpp

sort:
	nvcc -o debug/sort src/main.cpp src/sort.cu
	debug/sort input.txt 1000000 0 1000000 256 radix > output/sort.txt

test:
	nvcc -O3 -o debug/test src/main.cpp src/sort.cu
	debug/test input.txt 1000000 0 0 256 test > output/test.txt

gen:
	debug/textgen input_1M.txt 1000000
