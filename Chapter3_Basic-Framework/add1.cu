#include<stdio.h>           //头文件定义
#include<math.h>

const double EPSILON=1.0e-15;           //常量定义
const double a=1.23;
const double b=2.34;
const double c=3.57;
void __global__ add(const double *x,const double *y,double *z);         //CUDA核函数的声明
void check(const double*z,const int N);

int main()          //int main()
{
    const int N=100000000;
    const int M=sizeof(double)*N;
    double *h_x=(double*)malloc(M);
    double *h_y=(double*)malloc(M);
    double *h_z=(double*)malloc(M);

    for (int i = 0; i < N; i++)
    {
        /* code */
        h_x[i]=a;
        h_y[i]=b;
    }

    //在设备中也定义了三个数组,一般用d_x来表示设备内存,h_x来表示主机内存
    double *d_x,*d_y,*d_z;
    cudaMalloc((void**)&d_x,M);
    cudaMalloc((void**)&d_y,M);
    cudaMalloc((void**)&d_z,M);
    cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice);//主机到设备,前面第一个参数是dst,第二个参数是src
    cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_z,h_z,M,cudaMemcpyHostToDevice);

    const int block_size=128;   //线程块大小，即线程个数
    const int grid_size=N/block_size;   //网格大小，即线程块个数
    add<<<grid_size,block_size>>>(d_x,d_y,d_z);

    cudaMemcpy(h_z,d_z,M,cudaMemcpyDeviceToHost);
    check(h_z,N);
    
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add(const double*x,const double*y,double*z)
{
    const int n=blockDim.x*blockIdx.x+threadIdx.x;
    z[n]=x[n]+y[n];
}

void check(const double*z,const int N)
{
    bool has_error=false;
    for (int i = 0; i < N; i++)
    {
        if (fabs(z[i])-c>EPSILON)
        {
            has_error=true;
        }
    }    
    printf("%s\n",has_error?"Has errors":"No errors");
}
