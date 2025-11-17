#include<stdio.h>

__global__ void hello_from_gpu()
{
    const int bid =blockIdx.x;
    const int tid=threadIdx.x;
    printf("Hello World from block %d and  thread %d!\n",bid,tid);
}

int main()
{
    hello_from_gpu<<<2,4>>>();
    cudaDeviceSynchronize();    //同步主机设备，输出缓冲区
    return 0;
}