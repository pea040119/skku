#define MAX_STR_LEN 30
#define MIN_CHAR 65
#define MAX_CHAR 122



int gpu_check_sorted_arr(int block_size, int N, char str_arr_1[][MAX_STR_LEN], char str_arr_2[][MAX_STR_LEN]);

void bubble_sort(int N, char str_arr[][MAX_STR_LEN]);
void radix_sort(int N, char str_arr[][MAX_STR_LEN]);
int radixSort(char (*arr)[30], int N, char (*res)[30]);

void gpu_radix_sort(int block_size, int N, char str_arr[][MAX_STR_LEN]);