#include<stdio.h>
__global__ void helloWorldFromGPU(){
    printf("hello world from GPU!\n\n");
}

int main(void)
{
    helloWorldFromGPU<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

