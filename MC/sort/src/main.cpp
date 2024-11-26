#include<iostream>
#include<fstream>
#include<string.h>
#include<ctime>

using namespace std;

#include "sort.h"



int main(int argc, char* argv[]) {
    char temp_str[30];
    int i, j, N, pos, range, ret;
    string type;
    double elapsed;
    clock_t start, end;
    int block_size = 256;

    if(argc<7){
	    cout << "Usage: " << argv[0] << " filename number_of_strings pos range block_size type" << endl;
	    return 0;
    }

    cout << "[*] START INITIALIZING..." << endl;
    ifstream inputfile(argv[1]);
    if(!inputfile.is_open()){
	    cout << "Unable to open file" << endl;
	    return 0;
    }

    ret=sscanf(argv[2],"%d", &N);
    if(ret==EOF || N<=0){
	    cout << "Invalid number" << endl;
	    return 0;
    }

    ret=sscanf(argv[3],"%d", &pos);
    if(ret==EOF || pos<0 || pos>=N){
	    cout << "Invalid position" << endl;
	    return 0;
    }

    ret=sscanf(argv[4],"%d", &range);
    if(ret==EOF || range<0 || (pos+range)>=N){
	    cout << "Invalid range" << endl;
	    return 0;
    }

    ret=sscanf(argv[5],"%d", &block_size);
    if(ret==EOF || block_size<0){
	    cout << "Invalid range" << endl;
	    return 0;
    }

    type = argv[6];

    auto bubble_sort_arr = new char*[N];
    auto sort_arr = new char*[N];

    for(i=0; i<N; i++) {
        bubble_sort_arr[i] = new char[30];
        sort_arr[i] = new char[30];
    }

    for(i=0; i<N; i++) {
        inputfile >> temp_str;
        strcpy(bubble_sort_arr[i], temp_str);
        strcpy(sort_arr[i], temp_str);
    }

    inputfile.close();
    cout << "[+] END INITIALIZING" << endl;

    cout << "[*] START BUBBLE SORTING..." << endl;
    start = clock();
    bubble_sort(N, bubble_sort_arr);
    end = clock();
    elapsed = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "[+] END BUBBLE SORTING" << endl;
    cout << "[+] BUBBLE SORTING TIME ELAPSED: " << elapsed << "sec" << endl;

    cout << "[*] START "<< type <<" SORTING..." << endl;
    start = clock();
    gpu_radix_sort(block_size, N, sort_arr);
    end = clock();
    elapsed = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "[+] END "<< type <<" SORTING" << endl;
    cout << "[+] "<< type <<" SORTING TIME ELAPSED: " << elapsed << "sec" << endl;

    cout << "[*] START CHECK SORTED ARRAY..." << endl;
    int check = gpu_check_sorted_arr(block_size, N, bubble_sort_arr, sort_arr);
    cout << "[+] END CHECK SORTED ARRAY" << endl;
    cout << "[+] CHECK SORTED ARRAY: " << (check == 0 ? "TRUE" : "FALSE") << endl;
    cout << "[+] DIFFERENCE POSITION NUM: " << check << endl;

    cout<<"\n[+] SORT BY BUBBLE" << endl;
    for(i=pos; i<N && i<(pos+range); i++)
        cout<< i << ": " << bubble_sort_arr[i]<<endl;
    cout<<endl;

    cout<<"\n[+] SORT BY "<< type << endl;
    for(i=pos; i<N && i<(pos+range); i++)
        cout<< i << ": " << sort_arr[i]<<endl;
    cout<<endl;

    for(i=0; i<N; i++){
        delete[] bubble_sort_arr[i];
        delete[] sort_arr[i];
    }
    
    delete[] bubble_sort_arr;
    delete[] sort_arr;

    return 0;
}