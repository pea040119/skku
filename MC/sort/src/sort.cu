#include <cuda_runtime.h>

#include "sort.h"


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
    int result = 0;

    // 메모리 할당
    cudaMalloc((void**)&d_str_arr, N * sizeof(char*));
    cudaMalloc((void**)&d_sorted_arr, N * sizeof(char*));
    cudaMalloc((void**)&d_result, sizeof(int));

    // 초기화
    cudaMemcpy(d_str_arr, str_arr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_arr, sorted_arr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    // 커널 실행
    int block_num = (N + block_size - 1) / block_size;
    __check_sorted_arr_kernel<<<block_num, block_size>>>(N, d_str_arr, d_sorted_arr, d_result);

    // 결과 복사
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // 메모리 해제
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
        char ch = (pos < __strlen_kernel(str_arr[tid])) ? str_arr[tid][pos] : 0;
        atomicAdd(&count[ch], 1);
    }
}


void gpu_radix_sort(int block_size, int N, char **str_arr) {
}