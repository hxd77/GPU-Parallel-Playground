#include"error.cuh"
#include<stdio.h>

#ifdef USE_UP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS=20;
const int TILE_DIM=32;

void timing(const real*d_A,real*d_B,const int N, const int task);
__global__ void transpose1(const real*A,real*B,const int N);
__global__ void transpose2(const real*A,real*B,const int N);
void print_matrix(const int N,const real*A);

int main(int argc,char*argv[])
{
    if(argc!=2)
    {
        printf("Usage: %s N\n",argv[0]);
        eixt(1);
    }
    const int N=atoi(argv[1]);
    const int N2=N*N;
    const int M=sizeof(real)*N2;
    real*h_A=(real*)malloc(M);
    real*h_B=(real*)malloc(M);
    for(int n=0;n<N2;++n)
    {
        h_A[n]=n;
    }
    real*d_A,*d_B;
    CHECK(cudaMalloc((void**)&d_A,M));
    CHECK(cudaMalloc((void**)&d_B,M));
    CHECK(cudaMemcpy(d_A,h_A,M,cudaMemcpyHostToDevice));
    printf("\ntranspose with shared memory bank conflict:\n");
    timing(d_A,d_B,N,1);

    printf("\ntranspose with shared memory no bank conflict:\n");
    timing(d_A,d_B,N,2)
    CHECK(cudaMemcpy(h_B,d_B,M,cudaMemcpyDeviceToHost));

}

void timing(const real*d_A,real*d_B,const int N,const int task)
{
    const int grid_size_x=(N+TILE_DIM-1)/TILE_DIM;
    const int grid_size_y=(N+TILE_DIM-1)/TILE_DIM;
    dim3 grid_size(grid_size_x,grid_size_y);
    dim3 block_size(TILE_DIM,TILE_DIM);
    
    float t_sum=0;
    float t2_sum=0;
    for(int repeat=0;repeat<=NUM_REPEATS;repeat++)
    {
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch(task)
        {
            case 1:
                transpose1<<<grid_size,block_size>>>(d_A,d_B,N);
                break;
            case 2:
                transpose2<<<grid_size,block_size>>>(d_A,d_B,N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        printf("elapsed time = %g ms\n",elapsed_time);//%g表示科学计数法
        if(repeat>0)
        {
            t_sum+=elapsed_time;
            t2_sum+=elapsed_time*elapsed_time;
        }
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    const float t_ave=t_sum/NUM_REPEATS;
    const float t_err=sqrt(t2_sum/NUM_REPEATS-t_ave*t_ave);
    printf("Time = %g +- %g ms\n",t_ave,t_err);
}

__global__ void transpose1(const real*A,real*B,const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx=blockIdx.x*TILE_DIM;
    int by=blockIdx.y*TILE_DIM;
    
    int nx1=bx+threadIdx.x;
    int ny1=by+threadIdx.y;
    if(nx1<N&&ny1<N)
    {
        S[threadIdx.y][threadIdx.x]=A[ny1*N+nx1];
    }
    __syncthreads();    

    int nx2=bx+threadIdx.x;
    int ny2=by+threadIdx.y;
    if()
}