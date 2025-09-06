#include"error.cuh"
#include<stdio.h>

int main(int argc,char *argv[])
{
    int device_id=0;
    if(argc>1)device_id=atoi(argv[1]);//atoi将字符转化为数字
    
    CHECK(cudaSetDevice(device_id));    //选择使用的GPU

    cudaDeviceProp prop;    //定义结构体变量
    CHECK(cudaGetDeviceProperties(&prop,device_id));

    printf("Device id:                                  %d\n",device_id);
    printf("Device name:                                %s\n",prop.name);
    printf("Compute capability:                         %d.%d\n",prop.major,prop.minor);
    printf("Amount of global memory:                    %g GB\n",prop.totalGlobalMem/(1024.0*1024*1024));   //全局内存1 KB（千字节） = 1024 字节 1 MB（兆字节） = 1024 KB 1 GB（千兆字节） = 1024 MB
    printf("Amount of constant memory:                  %g KB\n",prop.totalConstMem/1024.0);
    printf("Maximum grid size:                          %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);   //x,y,z轴上的线程块大小
    printf("Maximum block size:                         %d %d %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    printf("Number of SMs:                              %d\n",prop.multiProcessorCount);/*流式多处理器（SM） 是 GPU 中执行计算任务的核心单元。一个 GPU 可能包含多个 SM，每个 SM 上有多个 CUDA 核心（CUDA Cores），这些核心可以并行执行不同的计算任务。*/
    printf("Maximum amount of shared memory per block:  %g KB\n",prop.sharedMemPerBlock/1024.0);//prop.sharedMemPerBlock 表示每个线程块（block）可以使用的 最大共享内存，单位为 字节（bytes）。
    printf("Maximum amount of shared memotry per SM:    %g KB\n",prop.sharedMemPerMultiprocessor/1024.0);
    printf("Maximum number of registers per block:      %d K\n",prop.regsPerBlock/1024.0);
    printf("Maximum number of registers per SM:         %d K\n",prop.regsPerMultiprocessor/1024);
    printf("Maximum number of threads per block:        %d \n",prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:           %d \n",prop.maxThreadsPerMultiProcessor);

    return 0;
}