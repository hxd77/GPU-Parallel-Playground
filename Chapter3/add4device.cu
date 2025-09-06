//版本一：有返回值的设备函数
double __device__ add1_device(const double x,const double y)
{
    return (x+y);   
}

void __global__ add1(const double*x,const double *y,double *z,const int N)
{
    const int n=blockDim.x*blockIdx.x+threadIdx.x;
    if (n<N)
    {
        /* code */
        z[n]=add1_device(x[n],y[n]);
    }
    
}

//版本二：用指针的设备函数
void __device__ add2_device(const double x,const double y,double *z)
{
    *z=x+y;
}

void __global__ add2(const double*x,const double*y,double*z,const int N)
{
    const int n=blockDim.x*blockIdx.x+threadIdx.x;
    if(n<N)
    {
        add2_device(x[n],y[n],&z[n]);
    }
}


//版本三：用引用（refeerence）的设备函数
void __device__ add3_device(const double x,const double y,double &z)
{
    z=x+y;
}

void __global__ add3(const double *x,const double *y,double*z,const int N )
{
    const int n=blockDim.x*blockIdx.x+threadIdx.x;
    if(n<N)
    {
        add3_device(x[n],y[n],z[n]);
    }
}