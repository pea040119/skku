#include <cuda_runtime.h>

#include "sort.h"


__device__ int __strncmp_kernel(const char *s1, const char *s2, size_t n) {
    while (n--) {
        if (*s1 != *s2) {
            return *(unsigned char *)s1 - *(unsigned char *)s2;
        }
        if (*s1 == '\0') {
            break;
        }
        s1++;
        s2++;
    }
    return 0;
}


__global__ void __check_sorted_arr_kernel(int N, char **strArr, char **sortedArr, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (__strncmp_kernel(strArr[idx], sortedArr[idx], 30) != 0) {
            atomicAdd(result, 1);
        }
    }
}

int check_sorted_arr(int N, char **strArr, char **sortedArr) {
    char **d_strArr, **d_sortedArr;
    int *d_result;
    int result = 0;

    // 메모리 할당
    cudaMalloc((void**)&d_strArr, N * sizeof(char*));
    cudaMalloc((void**)&d_sortedArr, N * sizeof(char*));
    cudaMalloc((void**)&d_result, sizeof(int));

    // 초기화
    cudaMemcpy(d_strArr, strArr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sortedArr, sortedArr, N * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    // 커널 실행
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    __check_sorted_arr_kernel<<<numBlocks, blockSize>>>(N, d_strArr, d_sortedArr, d_result);

    // 결과 복사
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // 메모리 해제
    cudaFree(d_strArr);
    cudaFree(d_sortedArr);
    cudaFree(d_result);

    return result;
}


void bubble_sort(int N, char **strArr) {
    char tmpStr[30];

    for(int i=1; i<N; i++)
    {
        for(int j=1; j<N; j++)
        {
            if(strncmp(strArr[j-1], strArr[j],30)>0)
            {
                strncpy(tmpStr, strArr[j-1], 30);
                strncpy(strArr[j-1], strArr[j], 30);
                strncpy(strArr[j], tmpStr, 30);
            }
        }
    }
}


__global__ void __p_r_sort_count_kernel(char **strArr, int *count, int N, int pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        char ch = (pos < strlen(strArr[tid])) ? strArr[tid][pos] : 0;
        atomicAdd(&count[ch], 1);
    }
}

void parallel_radix_sort(int N, char **strArr) {
}