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

void print_mem(){
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU memory: free = %zu bytes, total = %zu bytes\n", free_mem, total_mem);
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


__global__ void __gpu_radix_sort_init_arr_kernel(int *count, int *offset, int *preix_sum, int *N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *N)
        offset[tid] = 0;
    if (tid < MAX_CHAR-MIN_CHAR+1) {
        count[tid] = 0;
        preix_sum[tid] = 0;
    }

}

__global__ void __gpu_radix_sort_count_kernel(char **str_arr, int *count, int *offset, int *N, int *pos) {
    for(int tid=0; tid<*N; tid++) {
        int len = __strlen_kernel(str_arr[tid]);
        int index = (*pos < len) ? __char_to_index_kernel(str_arr[tid][*pos]) : 0;
        offset[tid] = count[index];
        atomicAdd(&count[index], 1);
    }
}

__global__ void __gpu_radix_sort_prefix_sum_kernel(int *count, int *prefix_sum) {
    for (int i=1; i<MAX_CHAR-MIN_CHAR+1; i++) {
        prefix_sum[i] = prefix_sum[i-1] + count[i-1];
    }
}

__global__ void __gpu_radix_sort_reorder_kernel(char **str_arr, char **output_arr, int *prefix_sum, int *offset, int *N, int *pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *N) {
        int len = __strlen_kernel(str_arr[tid]);
        int c_index = (*pos < len) ? __char_to_index_kernel(str_arr[tid][*pos]) : 0;
        int o_index = prefix_sum[c_index] + offset[tid];
        
        if (o_index < *N){
            // printf("o_index: %d\n", o_index);
            output_arr[o_index] = str_arr[tid];
        }
        else
            printf("ERROR: %d\ncount[c_index]: %d\toffset[tid]: %d\n", o_index, prefix_sum[c_index], offset[tid]);
    }
}

__global__ void __gpu_radix_sort_copy_arr_kernel(char **str_arr, char **output_arr, int *N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *N) {
        str_arr[tid] = output_arr[tid];
    }
}

__global__ void __gpu_radix_sort_print_str_arr(char **str_arr, int *N) {
    for (int i=0; i<*N; i++) {
        printf("%d: %s\n", i, str_arr[i]);
    }
}

__global__ void __gpu_radix_sort_print_int_arr(int *int_arr, int *N) {
    for (int i=0; i<*N; i++) {
        printf("%d: %i\n", i, int_arr[i]);
    }
}

void gpu_radix_sort(int block_size, int N, char **str_arr) {
    int *d_count, *d_prefix_sum, *d_offset, *d_N, *d_i;
    char **d_str_arr, **d_output;
    char *d_str;

    // print_mem();

    check_cuda_error(cudaMalloc((void**)&d_count, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_prefix_sum, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_str_arr, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_output, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_offset, N * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_N, sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_i, sizeof(int)));

    check_cuda_error(cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice));

    for (int i=0; i<N; i++) {
        check_cuda_error(cudaMalloc((void**)&d_str, MAX_STR_LEN * sizeof(char)));
        check_cuda_error(cudaMemcpy(d_str, str_arr[i], MAX_STR_LEN * sizeof(char), cudaMemcpyHostToDevice));
        check_cuda_error(cudaMemcpy(d_str_arr+i, &d_str, sizeof(char *), cudaMemcpyHostToDevice));
        check_cuda_error(cudaMemset(d_offset+i, 0, sizeof(int)));
    }
    for (int i=0; i<MAX_CHAR-MIN_CHAR+1; i++) {
        check_cuda_error(cudaMemset(d_count+i, 0, sizeof(int)));
        check_cuda_error(cudaMemset(d_prefix_sum+i, 0, sizeof(int)));
    }

    int block_num = (N + block_size - 1) / block_size;

    for (int i=MAX_STR_LEN-1;i>-1; i--) {
        // printf("i: %d\n", i);
        check_cuda_error(cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice));
        __gpu_radix_sort_init_arr_kernel<<<block_num, block_size>>>(d_count, d_offset, d_prefix_sum, d_N);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("init done\n");

        __gpu_radix_sort_count_kernel<<<1, 1>>>(d_str_arr, d_count, d_offset, d_N, d_i);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("count, offset done\n");
        
        __gpu_radix_sort_prefix_sum_kernel<<<1, 1>>>(d_count, d_prefix_sum);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("prefix sum done\n");

        __gpu_radix_sort_reorder_kernel<<<block_num, block_size>>>(d_str_arr, d_output, d_prefix_sum, d_offset, d_N, d_i);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("reorder done\n");


        __gpu_radix_sort_copy_arr_kernel<<<block_num, block_size>>>(d_str_arr, d_output, d_N);
        check_cuda_error(cudaDeviceSynchronize());
        // printf("copy done\n\n");
    }
    // printf("done\n");

    // __gpu_radix_sort_print_str_arr<<<1, 1>>>(d_str_arr, d_N);
    // check_cuda_error(cudaDeviceSynchronize());

    for (int i=0; i<N; i++) {
        check_cuda_error(cudaMemcpy(&d_str, d_str_arr + i, sizeof(char *), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaMemcpy(str_arr[i], d_str, MAX_STR_LEN * sizeof(char), cudaMemcpyDeviceToHost));
        check_cuda_error(cudaFree(d_str));
    }

    cudaFree(d_count);
    cudaFree(d_str_arr);
    cudaFree(d_output);
    cudaFree(d_offset);
    cudaFree(d_N);
    cudaFree(d_i);
}