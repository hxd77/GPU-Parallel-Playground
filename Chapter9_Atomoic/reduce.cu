#include<stdio.h>
#include"error.cuh"

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REAPEATS=10;
const int N=100000000;
const int M=sizeof(real)*N;
const int BLOCK_SIZE=128;


void timing(const real*d_x);

int main()
{
    real*h_x=(real*)malloc(M);
    for(int n=0;n<N;n++)
    {
        h_x[n]=1.23;
    }
    real*d_x;
    CHECK(cudaMalloc(&d_x,M));
    CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice));

    printf("\nusing atomiAdd:\n");
    timing(d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce(const real*d_x,real*d_y,const int N)
{
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    const int n=bid*blockDim.x+threadIdx.x;
    extern __shared__ real s_y[];
    s_y[tid]=(n<N)?d_x[n]:0.0;
    __syncthreads();    //执行该语句之前都完全执行了该语句前面的语句,该函数只针对一个线程块中

    for(int offset=blockDim.x>>1;offset>0;offset>>=1)
    {
        if(tid<offset)
        {
            s_y[tid]+=s_y[tid+offset];//0=0+64，1=1+65...63=63+64,0=0+32,1=1+33...31=31+32
        }
        __syncthreads();//不同线程块不影响，只保证一个线程块内的线程按照代码出现的顺序执行指令
    }

    if(tid==0)//tid=0的时候s_y[0]才是 当前 block 的所有线程的和；
    {
        atomicAdd(d_y,s_y[0]);//原子操作，保证每个每个线程的一气呵成
    }
}


real reduce(const real*d_x)
{
    const int grid_size=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    const int smem=sizeof(real)*BLOCK_SIZE;

    real h_y[1]={0};    //只用一个元素即可,但是要传指针,所以用数组
    real* d_y;
    CHECK(cudaMalloc(&d_y,sizeof(real)));
    CHECK(cudaMemcpy(d_y,h_y,sizeof(real),cudaMemcpyHostToDevice));

    reduce<<<grid_size,BLOCK_SIZE,smem>>>(d_x,d_y,N);

    CHECK(cudaMemcpy(h_y,d_y,sizeof(real),cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_y));

    return h_y[0];

}

void timing(const real*d_x)
{
    real sum=0;
    for(int repeat=0;repeat<NUM_REAPEATS;repeat++)
    {
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum=reduce(d_x);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        printf("Time = %g ms.\n",elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    printf("sum= %f.\n",sum);
}