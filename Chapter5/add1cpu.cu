#include "error.cuh"
#include<math.h>
#include<stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON=1.0e-15;
#else
    typedef float real;
    const real EPSILON=1.0e-6f; 
#endif

const int NUM_REPEATS=10;
const real a=1.23;
const real b=2.34;
const real c=3.57;
void add(const real*x,const real*y,real*z,const int N);
void check(const real*z, const int N);

int main(void)
{
    const int N=100000000;
    const int M=sizeof(real)*N;
    real*x=(real*)malloc(M);
    real*y=(real*)malloc(M);
    real*z=(real*)malloc(M);

    for(int n=0;n<N;++n)
    {
        x[n]=a;
        y[n]=b;
    }

    float t_sum=0;
    float t2_sum=0;
    for(int repeat=0;repeat<=NUM_REPEATS;++repeat)//测试11次，第一次计算时GPU可能处于预热状态，测得的时间往往偏大
    {
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));  //初始化start，stop事件
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));  //需要计时的代码块之前记录一个代表开始的事件
        cudaEventQuery(start);      //刷新队列

        add(x,y,z,N);

        CHECK(cudaEventRecord(stop));    //需要计时的代码块之后记录一个代表结束的事件
        CHECK(cudaEventSynchronize(stop));  //让主机等待事件stop记录完毕
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));  //函数计算start和stop这两个事件之间的时间差(单位是ms)输出到屏幕
        printf("Time = %g ms.\n",elapsed_time);

        if(repeat>0)    //只算后面10次
        {
            t_sum+=elapsed_time;
            t2_sum+=elapsed_time*elapsed_time;
        }

        CHECK(cudaEventDestroy(start));     //销毁start和stop两个CUDA事件
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave=t_sum/NUM_REPEATS;
    const float t_err=sqrt(t2_sum/NUM_REPEATS-t_ave*t_ave);//方差 = 平方的均值 − 均值的平方 标准差 = 方差的平方根
    printf("Time = %g +- %g ms.\n",t_ave,t_err);

    check(z,N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real*x,const real*y,real*z,const int N)
{
    for(int n=0;n<N;++n)
    {
        z[n]=x[n]+y[n];
    }
}

void check(const real*z,const int N)
{
    bool has_error=false;
    for(int n=0;n<N;++n)
    {
        if(fabs(z[n]-c)>EPSILON)
        {
            has_error=true;
        }
    }
    printf("%s\n",has_error? "Has errors":"No errors");
}

