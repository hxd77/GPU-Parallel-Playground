#include<stdio.h>

__global__ void hello_from_gpu()
{
    const int b=blockIdx.x;//该变量指定一个线程在一个网格中的线程块指标，取值从0-gridDim.x-1
    const int tx=threadIdx.x; //该变量指定一个线程在一个线程块中的线程指标，取值从0-blockDim.x-1
    const int ty=threadIdx.y;
    printf("Hello World from block - %d and thead - (%d,%d)!\n",b,tx,ty);
}

int main()
{
    const dim3 block_size(2,4);//线程块大小为2*4*1,z默认为1
    hello_from_gpu<<<1,block_size>>>();//网格大小为1，线程块大小为2*4*1
    cudaDeviceSynchronize();
    return 0;
}