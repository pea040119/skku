#include <cuda_runtime.h>
#include<iostream>

using namespace std;

#include "sort.h"

#define MAX_STR_LEN 30
#define MIN_CHAR 65
#define MAX_CHAR 122



bool check_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(status) << endl;
        return false;
    }
    return true;
}


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



__global__ void __check_sorted_arr_kernel(int *N, char **str_arr, char **sorted_arr, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *N) {
        if (__strncmp_kernel(*(str_arr+idx), *(sorted_arr+idx), MAX_STR_LEN) != 0) {
            atomicAdd(result, 1);
        }
    }
}

__global__ void __test(char *str) {
    printf("%c\n", *(str+1));
}

int gpu_check_sorted_arr(int block_size, int N, char **str_arr, char **sorted_arr) {
    if (N <= 0 || block_size <= 0 || str_arr == NULL || sorted_arr == NULL) 
        return -1;

    char **d_str_arr, **d_sorted_arr, *d_str;
    int *d_result, *d_N;
    int result;

    check_cuda_error(cudaMalloc((void**)&d_str_arr, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_sorted_arr, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_result, sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_N, sizeof(int)));
    for(int i=0; i<N; i++) {
        check_cuda_error(cudaMalloc((void**)&d_str, sizeof(char) * MAX_STR_LEN));
        check_cuda_error(cudaMemcpy(d_str, str_arr[i], sizeof(char) * MAX_STR_LEN, cudaMemcpyHostToDevice));
        check_cuda_error(cudaMemcpy(d_str_arr+i, &d_str, sizeof(char*), cudaMemcpyHostToDevice));
        check_cuda_error(cudaMalloc((void**)&d_str, sizeof(char) * MAX_STR_LEN));
        check_cuda_error(cudaMemcpy(d_str, sorted_arr[i], sizeof(char) * MAX_STR_LEN, cudaMemcpyHostToDevice));
        check_cuda_error(cudaMemcpy(d_sorted_arr+i, &d_str, sizeof(char*), cudaMemcpyHostToDevice));
    }
    check_cuda_error(cudaMemset(d_result, 0, sizeof(int)));
    check_cuda_error(cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice));

    int block_num = (N + block_size - 1) / block_size;
    __check_sorted_arr_kernel<<<block_num, block_size>>>(d_N, d_str_arr, d_sorted_arr, d_result);
    check_cuda_error(cudaDeviceSynchronize());

    check_cuda_error(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // for (int i=0; i<N; i++) {
    //     check_cuda_error(cudaFree(*(str_arr+i)));
    //     check_cuda_error(cudaFree(*(sorted_arr+i)));
    // }
    cudaFree(d_str_arr);
    cudaFree(d_sorted_arr);
    cudaFree(d_result);
    cudaFree(d_N);

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
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
}

void gpu_radix_sort(int block_size, int N, char **str_arr) {
    int *d_count;
    char **d_str_arr, **d_output;
    char *d_str;

    check_cuda_error(cudaMalloc((void**)&d_count, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_str_arr, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_output, N * sizeof(char *)));

    for (int i=0; i<N; i++) {
        check_cuda_error(cudaMalloc((void**)&d_str, MAX_STR_LEN * sizeof(char)));
        check_cuda_error(cudaMemcpy(d_str, str_arr[i], MAX_STR_LEN * sizeof(char), cudaMemcpyHostToDevice));
        check_cuda_error(cudaMemcpy(d_str_arr+i, &d_str, sizeof(char *), cudaMemcpyHostToDevice));
    }

    int block_num = (N + block_size - 1) / block_size;

    for (int i=0; i<MAX_STR_LEN; i++) {
        // printf("i: %d\t", i);
        for (int j=0; j<(MAX_CHAR - MIN_CHAR + 1); j++) {
            check_cuda_error(cudaMemset(d_count+j, 0, sizeof(int)));
        }
        // printf("memset done\t");
        __gpu_radix_sort_count_kernel<<<block_num, block_size>>>(d_str_arr, d_count, N, i);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("count done\t");
        
        __gpu_radix_sort_prefix_sum_kernel<<<1, 1>>>(d_count, N);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("prefix sum done\t");


        __gpu_radix_sort_reorder_kernel<<<1, MAX_CHAR-MIN_CHAR+1>>>(d_str_arr, d_output, d_count, N, i);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("\n");

    }

    // cudaMemcpy(str_arr, d_output, N * sizeof(char*), cudaMemcpyDeviceToHost);

    cudaFree(d_count);
    cudaFree(d_str_arr);
    cudaFree(d_output);
}