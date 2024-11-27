all:
	g++ -g -O2 -o seqsort sequential_sort.cpp
	g++ -g -o textgen text_gen.cpp
	nvcc -O3 -o debug/sort src/main.cpp src/sort.cu

sort:
	nvcc -O3 -o debug/sort src/main.cpp src/sort.cu
	debug/sort input.txt 1000000 0 1000000 256 gpu_radix > output/sort.txt

test:
	nvcc -O3 -o debug/test src/main.cpp src/sort.cu
	debug/test input.txt 1000000 0 0 256 gpu_radix > output/test.txt

bubble:
	nvcc -o debug/sort src/main.cpp src/sort.cu
	debug/sort input.txt 1000000 0 1000000 256 bubble > output/bubble.txt

radix:
	nvcc -o debug/sort src/main.cpp src/sort.cu
	debug/sort input.txt 1000000 0 1000000 256 radix > output/radix.txt

gen:
	debug/textgen input_1M.txt 1000000
