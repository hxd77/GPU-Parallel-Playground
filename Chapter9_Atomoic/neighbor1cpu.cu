#include<stdio.h>
#include<cmath>
#include"error.cuh"
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
using namespace std;

#ifdef USE_DP
    typedef double real;
#else 
    typedef float real;
#endif

int N;      //原子总数
const int NUM_REAPEATS=10;  //时间数
const int MN=10;    //每个原子的最大邻居数
const real cutoff=1.9;  //以Angstrom为单位
const real cutoff_square=cutoff*cutoff;

void read_xy(std::vector<real>& x, std::vector<real>& y);
void timing(int*NN,int*NL,vector<real>x,vector<real>y);
void print_neighbor(const int *NN,const int *NL);

int main()
{
    vector<real>x,y;
    read_xy(x,y);//先读取数据
    N=x.size();
    int *NN=(int*)malloc(N*sizeof(int));//NN数组长度为N
    int *NL=(int*)malloc(N*MN*sizeof(int));

    timing(NN,NL,x,y);
    print_neighbor(NN,NL);

    free(NN);
    free(NL);
    return 0;
}

void read_xy(vector<real>&v_x,vector<real>&v_y)
{
    ifstream infile("xy.txt");//读取一个输入文件流对象
    string line,word;
    if(!infile)
    {
        cout<<"Cannot open xy.txt"<<endl;
        exit(1);
    }
    while(getline(infile,line))//getline读取文件中的每一行
    {
        istringstream words(line);//istringstream把读取的每一行封装成一个输入字符串流对象words
        if(line.length()==0)
        {
            continue;
        }
        for(int i=0;i<2;i++)
        {
            if(words>>word)//因为就只有两个元素x和y坐标，从字符串流 words 中读取一个用空格分隔的单词或数字并存入变量word中，如果读取成功就为True，读取失败则为False
            {
                if(i==0)//第一个元素放入x数组
                {
                    v_x.push_back(stod(word));//stod=string to double 
                }
                if(i==1)//第二个元素放入y数组
                {
                    v_y.push_back(stod(word));
                }
            }
            else
            {
                cout<<"Error for reading xy.txt"<<endl;
                exit(1);
            }
        }

    }
    infile.close();
}

void find_neighbor(int *NN,int *NL,const real*x,const real*y)
{
    //NN[n]是第n个粒子的邻居个数
    //NL[n*MN+k]是第n个粒子的第k个邻居的指标
    
    for(int n=0;n<N;n++)
    {
        NN[n]=0;    //初始化所有粒子的邻居数量为0
    }
    //遍历所有粒子对(n1,n2)
    for(int n1=0;n1<N;++n1)
    {
        real x1=x[n1];
        real y1=y[n1];
        for(int n2=n1+1;n2<N;n2++)
        {
            real x12=x[n2]-x1;//坐标之差
            real y12=y[n2]-y1;
        
            real distance_square=x12*x12+y12*y12;
            if(distance_square<cutoff_square)
           {
                NL[n1*MN+NN[n1]++]=n2;
                NL[n2*MN+NN[n2]++]=n1;

            }
        }
    }
}
void timing(int*NN,int*NL,vector<real>x,vector<real>y)
{
    for(int repeat=0;repeat<NUM_REAPEATS;repeat++)
    {
        cudaEvent_t start,stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        
        find_neighbor(NN,NL,x.data(),y.data());
        
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time,start,stop));
        cout<<"Time = "<<elapsed_time<< " ms ."<<endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int*NN,const int *NL)
{
    ofstream outfile("neighbor.txt");
    if(!outfile)//outfile为空
    {
        cout<<"Cannot open neighbor.txt "<<endl;
    }
    for(int n=0;n<N;++n)
    {
        if(NN[n]>MN)//如果第n个粒子的邻居大于10
        {
            cout<<"Error: MN is too small. "<<endl;
            exit(1);
        }
        outfile<<NN[n];//先输出有几个邻居
        for(int k=0;k<MN;k++)
        {
            if(k<NN[n])
            {
                outfile<< " "<<NL[n*MN+k];//然后输出第n个粒子的第k个邻居的坐标
            }
            else
            {
                outfile << " NaN";  //否则输出NaN
            }
        }
        outfile<<endl;
    }
    outfile.close();
}