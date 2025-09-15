# CUDA第1章：GPU和CUDA简介

## 1.1 GPU简介

GPU的意思是`图形处理单元`，通常与CPU（中央处理器）相提并论。典型的CPU有几个相对较快的内核，而典型的GPU有数百或数千个相对较慢的内核。在CPU中，更多的晶体管专门用于缓存和控制；在GPU中，更多的晶体管专门用于数据处理。

 GPU计算是`异构计算`。它涉及CPU和GPU，通常分别称为`主机`和`设备`。CPU和非嵌入式GPU都有自己的DRAM（动态随机存取存储器），它们通常通过PCle（外围组件互连快速）总线连接。

我们只考虑Nvidia的GPU，因为CUDA变成只支持这些GPU。有几个系列的Nvidia GPU：

+ Tesla系列：适合科学计算，但价格昂贵。
+ GeForce系列：更便宜但不太专业。
+ Quadro系列：介于上述两者之间。
+ Jetson系列：嵌入式设备

每个GPU都有一个版本号`X.Y`来表示其**计算能力**。这里，`X`是主要版本号，`Y`是次要版本号。主要版本号对应于主要的GPU架构，也以一位著名的科学家的名字命名。请参见下表。

| 主要计算能力 | 架构代号 | 发行年份 |
| :----------: | :------: | :------: |
|    `X=1`     |  Tesla   |   2006   |
|    `x=2`     |  Feimi   |   2010   |
|    `X=3`     |  Kepler  |   2012   |
|    `X=5`    | Maxwell  |   2014   |
|    `X=6`    |  Pascal  |   2016   |
|    `X=7`    | Volta | 2017 |
|    `X.Y=7.5`    | Turing | 2018 |
|  `X=8`  | Ampere | 2020 |

比Pascal更旧的GPU将很快被弃用。我们将重点关注不比Pascal更旧的GPU。

GPU的计算能力与其性能没有直接关系。下表列出了有关一些选定GPU性能的主要指标。

|    GPU     | 计算能力 | 显存容量/GB | 内存带宽/（GB/s） | 双精度峰值FLOPS | 单精度峰值FLOPS |
| :--------: | :------: | :---------: | :---------------: | :-------------: | :-------------: |
| Tesla P100 |   6.0    |    16 GB    |     732 GB/s      |   4.7 TFLOPS    |   9.3 TFLOPS    |
| Tesla V100 | 7.0 | 32 GB | 900 GB/s | 7 TFLOPS |14 TFLOPS|
| GeForce RTX 2070 | 7.5 | 8 GB | 448 GB/s | 0.2 TFLOPS |6.5 TFLOPS|
| GeForce RTX 2080ti | 7.5 | 11 GB | 732 GB/s | 0.4 TFLOPS |13 TFLOPS|

我们注意到，GeForce GPU的双精度性能仅为其单精度性能的1/32。

## 1.2 CUDA简介

GPU计算的工具有几种，包括CUDA、OpenCL和OpenACC，但我们在这里只考虑了CUDA。我们也只考虑基于C++的CUDA，简称CUDA C++。我们不会考虑CUDA FORTRAN。

CUDA为开发人员提供了两个API（应用程序编程接口）：**CUDA驱动程序API**和**CUDA运行时API**。CUDA驱动程序API更基础（低级）且更灵活。CUDA运行时API基于CUDA驱动程序API构建，更易于使用。我们只考虑CUDA运行时API。



