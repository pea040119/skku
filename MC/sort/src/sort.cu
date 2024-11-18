#include <cuda_runtime.h>
#include<iostream>

using namespace std;

#include "sort.h"

#define MAX_STR_LEN 30
#define MIN_CHAR 65
#define MAX_CHAR 122



__device__ int __strncmp_kernel(const char *str_1, const char *str_2, size_t n) {
    while (n--) {
        if (*str_1 != *str_2) {
            return *(unsigned char *)str_1 - *(unsigned char *)str_2;
        }
        if (*str_1 == '\0') {
            break;
        }
        str_1++;
        str_2++;
    }
    return 0;
}

__device__ int __strlen_kernel(const char *str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ int __char_to_index_kernel(char ch){
    return (int)ch - MIN_CHAR+1;
}

__device__ char __index_to_char_kernel(int n){
    return (char)(n + MIN_CHAR-1);
}



__global__ void __check_sorted_arr_kernel(int N, char **str_arr, char **sorted_arr, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (__strncmp_kernel(str_arr[idx], sorted_arr[idx], 30) != 0) {
            atomicAdd(result, 1);
        }
    }
}

int gpu_check_sorted_arr(int block_size, int N, char **str_arr, char **sorted_arr) {
    if (N <= 0 || block_size <= 0 || str_arr == NULL || sorted_arr == NULL) 
        return -1;

    char **d_str_arr, **d_sorted_arr;
    int *d_result;
    int result = 12;

    cudaMalloc((void**)&d_str_arr, N * sizeof(char*));
    cudaMalloc((void**)&d_sorted_arr, N * sizeof(char*));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_str_arr, str_arr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_arr, sorted_arr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    int block_num = (N + block_size - 1) / block_size;
    __check_sorted_arr_kernel<<<block_num, block_size>>>(N, d_str_arr, d_sorted_arr, d_result);

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_str_arr);
    cudaFree(d_sorted_arr);
    cudaFree(d_result);

    return result;
}


void bubble_sort(int N, char **str_arr) {
    char temp_str[30];

    for(int i=1; i<N; i++)
    {
        for(int j=1; j<N; j++)
        {
            if(strncmp(str_arr[j-1], str_arr[j],30)>0)
            {
                strncpy(temp_str, str_arr[j-1], 30);
                strncpy(str_arr[j-1], str_arr[j], 30);
                strncpy(str_arr[j], temp_str, 30);
            }
        }
    }
}


__global__ void __gpu_radix_sort_count_kernel(char **str_arr, int *count, int N, int pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int index = (pos < __strlen_kernel(str_arr[tid])) ? __char_to_index_kernel(str_arr[tid][pos]) : 0;
        atomicAdd(&count[index], 1);
    }
}

__global__ void __gpu_radix_sort_prefix_sum_kernel(int *count, int N) {
    for (int i=1; i<MAX_CHAR-MIN_CHAR+1; i++) {
        count[i] += count[i-1];
    }
}

__global__ void __gpu_radix_sort_reorder_kernel(char **str_arr, char **output_arr, int *count, int N, int pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int index = (pos < __strlen_kernel(str_arr[tid])) ? __char_to_index_kernel(str_arr[tid][pos]) : 0;
        int new_index = atomicAdd(&count[index], 1);
        output_arr[new_index] = str_arr[tid];
    }
}

void gpu_radix_sort(int block_size, int N, char **str_arr) {
    int *d_count;
    char **d_str_arr, **d_output;

    cudaMalloc((void**)&d_count, MAX_CHAR-MIN_CHAR+1 * sizeof(int));
    cudaMalloc((void**)&d_str_arr, N * sizeof(char*));
    cudaMalloc((void**)&d_output, N * sizeof(char*));

    cudaMemcpy(d_str_arr, str_arr, N * sizeof(char*), cudaMemcpyHostToDevice);

    int block_num = (N + block_size - 1) / block_size;

    // cout << "start sorting" << endl;

    for (int i=0; i<MAX_STR_LEN; i++) {
        cudaMemset(d_count, 0, (MAX_CHAR - MIN_CHAR + 1) * sizeof(int));
        __gpu_radix_sort_count_kernel<<<block_num, block_size>>>(d_str_arr, d_count, N, i);
        // cout << "counting done" << endl;
        
        __gpu_radix_sort_prefix_sum_kernel<<<1, 1>>>(d_count, N);
        // cout << "prefix sum done" << endl;

        __gpu_radix_sort_reorder_kernel<<<block_num, block_size>>>(d_str_arr, d_output, d_count, N, i);
        // cout << "reordering done" << endl;
        cudaMemcpy(d_str_arr, d_output, N * sizeof(char*), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(str_arr, d_str_arr, N * sizeof(char*), cudaMemcpyDeviceToHost);

    cudaFree(d_count);
    cudaFree(d_str_arr);
    cudaFree(d_output);
}