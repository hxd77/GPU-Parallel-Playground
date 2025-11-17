#include "error.cuh"
#include<stdio.h>
__device__ int d_x=1;       //静态全局内存变量
__device__ int d_y[2];      //静态全局内存变量固定长度的数组

void __global__ my_kernel(void)
{
    d_y[0]+=d_x;
    d_y[1]+=d_x;
    printf("d_x = %d, d_y[0]=%d, d_y[1]= %d.\n", d_x,d_y[0],d_y[1]);
}

int main()
{
    int h_y[2]={10,20};
    CHECK(cudaMemcpyToSymbol(d_y,h_y,sizeof(int)*2));//将主机数组h_y中的数据复制到静态全局内存数组d_y
    
    my_kernel<<<1,1>>>();
    CHECK(cudaDeviceSynchronize()); //同步主机与设备，输出缓冲区
    
    CHECK(cudaMemcpyFromSymbol(h_y,d_y,sizeof(int)*2));   //将静态全局内存数组d_y中的数据复制到主机数组h_y
    printf("h_y[0]=%d,h_y[1]=%d.\n",h_y[0],h_y[1]);

    return 0;
}