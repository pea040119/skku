#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

using namespace std;

#include "sort.h"



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
    while (str[len] != '\0')
        len++;
    return len;
}

__device__ void __strncpy_kernel(char *dest, const char *src, size_t n) {
    for (int i=0; i<n; i++) {
        dest[i] = src[i];
        if (src[i] == '\0')
            break;
    }
}

__device__ int __char_to_index_kernel(char ch){
    return (int)ch - MIN_CHAR+1;
}

__device__ char __index_to_char_kernel(int n){
    return (char)(n + MIN_CHAR-1);
}



__global__ void __check_sorted_arr_kernel(int *N, char str_arr_1[][MAX_STR_LEN], char str_arr_2[][MAX_STR_LEN], int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < *N) {
        if (__strncmp_kernel(str_arr_1[tid], str_arr_2[tid], MAX_STR_LEN) != 0)
            atomicAdd(result, 1);
    }
    __syncthreads();
}

__global__ void __test(char *str) {
    printf("%c\n", *(str+1));
}

int gpu_check_sorted_arr(int block_size, int N, char str_arr_1[][MAX_STR_LEN], char str_arr_2[][MAX_STR_LEN]) {
    if (N <= 0 || block_size <= 0 || str_arr_1 == NULL || str_arr_2 == NULL) 
        return -1;

    char (*d_str_arr_1)[MAX_STR_LEN], (*d_str_arr_2)[MAX_STR_LEN];
    int *d_result, *d_N;
    int result;
    
    check_cuda_error(cudaMalloc((void**)&d_str_arr_1, N * sizeof(char *)));
    check_cuda_error(cudaMalloc((void**)&d_str_arr_2, N * sizeof(char *)));

    check_cuda_error(cudaMemcpy(d_str_arr_1, str_arr_1, N * sizeof(char)*MAX_STR_LEN, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_str_arr_2, str_arr_2, N * sizeof(char)*MAX_STR_LEN, cudaMemcpyHostToDevice));

    check_cuda_error(cudaMalloc((void**)&d_result, sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_N, sizeof(int)));
    
    check_cuda_error(cudaMemset(d_result, 0, sizeof(int)));
    check_cuda_error(cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice));

    int block_num = (N + block_size - 1) / block_size;
    __check_sorted_arr_kernel<<<block_num, block_size>>>(d_N, d_str_arr_1, d_str_arr_2, d_result);

    check_cuda_error(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_str_arr_1);
    cudaFree(d_str_arr_2);
    cudaFree(d_result);
    cudaFree(d_N);

    return result;
}


void bubble_sort(int N, char str_arr[][MAX_STR_LEN]) {
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


void radix_sort(int N, char str_arr[][MAX_STR_LEN]) {
    cout << "test" << endl;
    int count[MAX_CHAR-MIN_CHAR+1], prefix_sum[MAX_CHAR-MIN_CHAR+1], offset[N];
    char output_arr[N][MAX_STR_LEN];
    cout << "test" << endl;
    
    memset(count, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int));
    memset(prefix_sum, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int));
    memset(offset, 0, N * sizeof(int));

    cout << "test" << endl;
    for (int i=MAX_STR_LEN-1; i>-1; i--) {
        // printf("i: %d\n", i);
        memset(count, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int));
        memset(prefix_sum, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int));
        memset(offset, 0, N * sizeof(int));
        for (int j=0; j<N; j++) {
            int len = strlen(str_arr[j]);
            int index = (i < len) ? str_arr[j][i] - MIN_CHAR+1 : 0;
            offset[j] = count[index];
            count[index]++;
        }

        for (int j=1; j<MAX_CHAR-MIN_CHAR+1; j++)
            prefix_sum[j] = prefix_sum[j-1] + count[j-1];

        for (int j=0; j<N; j++) {
            int len = strlen(str_arr[j]);
            int c_index = (i < len) ? str_arr[j][i] - MIN_CHAR+1 : 0;
            int o_index = prefix_sum[c_index] + offset[j];
            strncpy(output_arr[o_index], str_arr[j], MAX_STR_LEN);
        }

        for (int j=0; j<N; j++)
            strncpy(str_arr[j], output_arr[j], MAX_STR_LEN);
    }
}


int radixSort(char (*arr)[30], int N, char (*res)[30]) {
  for (int i = 29; i >= 0; i--) {
    int hist[128] = {0};

    for (int j = 0; j < N; j++) {
      char c = (i < strlen(arr[j])) ? arr[j][i] : 0;
      hist[c]++;
    }

    int prev = 0;
    for (int j = 0; j < 128; j++) {
      int temp = hist[j];
      hist[j] = prev;
      prev += temp;
    }

    for (int j = 0; j < N; j++) {
      char c = (i < strlen(arr[j])) ? arr[j][i] : 0;
      strncpy(res[hist[c]], arr[j], 30);
      hist[c]++;
    }

    for (int j = 0; j < N; j++) {
      strncpy(arr[j], res[j], 30);
    }
  }
  
  return 0;
}


__global__ void __gpu_radix_sort_init_arr_kernel(int *count, int *offset, int *preix_sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        offset[tid] = 0;
    if (tid < MAX_CHAR-MIN_CHAR+1) {
        count[tid] = 0;
        preix_sum[tid] = 0;
    }
    __syncthreads();
}

__global__ void __gpu_radix_sort_count_kernel(char str_arr[][MAX_STR_LEN], int *count, int *offset, int N, int pos) {
    for(int tid=0; tid<N; tid++) {
        int len = __strlen_kernel(str_arr[tid]);
        int index = (pos < len) ? __char_to_index_kernel(str_arr[tid][pos]) : 0;
        offset[tid] = count[index];
        atomicAdd(&count[index], 1);
    }
    __syncthreads();
}

__global__ void __gpu_radix_sort_prefix_sum_kernel(int *count, int *prefix_sum) {
    for (int i=1; i<MAX_CHAR-MIN_CHAR+1; i++)
        prefix_sum[i] = prefix_sum[i-1] + count[i-1];
    __syncthreads();
}

__global__ void __gpu_radix_sort_reorder_kernel(char str_arr[][MAX_STR_LEN], char output_arr[][MAX_STR_LEN], int *prefix_sum, int *offset, int N, int pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int len = __strlen_kernel(str_arr[tid]);
        int c_index = (pos < len) ? __char_to_index_kernel(str_arr[tid][pos]) : 0;
        int o_index = prefix_sum[c_index] + offset[tid];
        
        if (o_index < N)
            __strncpy_kernel(output_arr[o_index], str_arr[tid], MAX_STR_LEN);
        else
            printf("ERROR: %d\ncount[c_index]: %d\toffset[tid]: %d\n", o_index, prefix_sum[c_index], offset[tid]);
    }
    __syncthreads();
}

__global__ void __gpu_radix_sort_copy_arr_kernel(char str_arr[][MAX_STR_LEN], char output_arr[][MAX_STR_LEN], int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        __strncpy_kernel(str_arr[tid], output_arr[tid], MAX_STR_LEN);
    __syncthreads();
}

__global__ void __gpu_radix_sort_print_str_arr(char str_arr[][MAX_STR_LEN], int N) {
    for (int i=0; i<N; i++)
        printf("%d: %s\n", i, str_arr[i]);
    __syncthreads();
}

__global__ void __gpu_radix_sort_print_int_arr(int *int_arr, int N) {
    for (int i=0; i<N; i++)
        printf("%d: %i\n", i, int_arr[i]);
    __syncthreads();
}

__global__ void __gpu_radix_sprt_print_str_p_arr(char char_arr[][30], int N) {
    for (int i=0; i<N; i++)
        printf("%d: %p\n", i, char_arr[i]);
    __syncthreads();
}

void gpu_radix_sort(int block_size, int N, char str_arr[][MAX_STR_LEN]) {
    int *d_count, *d_prefix_sum, *d_offset;
    char (*d_str_arr)[30], (*d_output)[30];
    clock_t start, end;

    check_cuda_error(cudaMalloc((void**)&d_count, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_prefix_sum, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMalloc((void**)&d_offset, N * sizeof(int)));

    check_cuda_error(cudaMemset(d_count, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMemset(d_prefix_sum, 0, (MAX_CHAR-MIN_CHAR+1) * sizeof(int)));
    check_cuda_error(cudaMemset(d_offset, 0, N * sizeof(int)));

    check_cuda_error(cudaMalloc((void**)&d_str_arr, N * sizeof(char)* MAX_STR_LEN));
    check_cuda_error(cudaMalloc((void**)&d_output, N * sizeof(char)* MAX_STR_LEN));

    start = clock();
    check_cuda_error(cudaMemcpy(d_str_arr, str_arr, N * sizeof(char) * MAX_STR_LEN, cudaMemcpyHostToDevice));
    end = clock();
    printf("copy H to D: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
    int block_num = (N + block_size - 1) / block_size;

    start = clock();
    for (int i=MAX_STR_LEN-1;i>-1; i--) {
        __gpu_radix_sort_init_arr_kernel<<<block_num, block_size>>>(d_count, d_offset, d_prefix_sum, N);
        __gpu_radix_sort_count_kernel<<<1, 1>>>(d_str_arr, d_count, d_offset, N, i);
        __gpu_radix_sort_prefix_sum_kernel<<<1, 1>>>(d_count, d_prefix_sum);
        __gpu_radix_sort_reorder_kernel<<<block_num, block_size>>>(d_str_arr, d_output, d_prefix_sum, d_offset, N, i);
        __gpu_radix_sort_copy_arr_kernel<<<block_num, block_size>>>(d_str_arr, d_output, N);
    }
    end = clock();
    printf("sort: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    start = clock();
    check_cuda_error(cudaDeviceSynchronize());
    end = clock();
    printf("sync: %f\n", (double)(end-start)/CLOCKS_PER_SEC);
    
    start = clock();
    check_cuda_error(cudaMemcpy(str_arr, d_str_arr, N * sizeof(char) * MAX_STR_LEN, cudaMemcpyDeviceToHost));
    end = clock();
    printf("copy D to H: %f\n", (double)(end-start)/CLOCKS_PER_SEC);

    cudaFree(d_count);
    cudaFree(d_str_arr);
    cudaFree(d_output);
    cudaFree(d_offset);
    cudaFree(d_prefix_sum);
}