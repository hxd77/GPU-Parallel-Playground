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

还有更多CUDA版本，也可以表示为`X.Y`。但是并不等同于GPU的计算能力。下表列出了一些最新的CUDA版本和支持的计算功能。

|   CUDA 版本    |               支持的GPU               |
| :------------: | :-----------------------------------: |
|   CUDA 11.0    | 计算能力 3.5-8.0 （Kepler to Ampere） |
| CUDA 10.0-10.2 | 计算能力 3.0-7.5 （Kepler to Turing） |
|  CUDA 9.0-9.2  |  计算能力 3.0-7.2 (Kepler to Volta)   |
|    CUDA 8.0    | 计算能力 2.0-6.2 （Fermi to Pascal）  |

## 1.3 安装GPU

对于Linux，请查看此手册：https://docs.nvidia.com/cuda/cuda-installation-guide-linux

对于 Windows，需要同时安装 CUDA 和 Visual Studio：

+ 安装 Visual Studio。转到 https://visualstudio.microsoft.com/free-developer-offers/ 并下载免费的 Visual Studio（社区版）。出于本书的目的，我们只需要在 Visual Studio 的许多组件中使用 `C++ 安装桌面开发 `。

- 安装 CUDA。转到 https://developer.nvidia.com/cuda-downloads 并选择 Windows CUDA 版本并安装它。您可以选择支持您的 GPU 的最高版本。

  安装 Visual Studio 和 CUDA 后（`ProgramData` 文件夹可能被隐藏，您可以启用以显示它），转到以下文件夹

```
 C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery 
```

，然后使用 Visual Studio 打开解决方案 `deviceQuery_vs2019.sln`。然后构建解决方案并运行可执行文件。如果您在输出末尾看到 `Result = PASS`，那么恭喜您！如果遇到问题，可以仔细查看手册：https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows。

在本书中，我们不会使用 Visual Studio IDE 来开发 CUDA 程序。相反，我们使用**命令行** （在 Linux 中称为**终端** ，在 Windows 中称为**命令提示符** ），如果需要，还可以使用 `make` 程序。在 Windows 中，要使 MSVC（Microsoft Visual C++ Compiler）`cl.exe` 可用，可以按照以下步骤打开命令提示符：

```
Windows start -> Visual Studio 2019 -> x64 Native Tools Command Prompt for VS 2019
```

在某些情况下，我们需要拥有管理员权限，这可以通过右键单击 `x64 Native Tools Command Prompt for VS 2019` 并选择`更多 `，然后以`管理员身份运行`来实现。



## 1.4 使用 `nvidia-smi` 程序

安装 CUDA 后，应该能够从命令行使用程序（可执行文件）`nvidia-smi`（**Nvidia 的系统管理界面** ）。只需输入此程序的名称：

```
$ nvidia-smi
```

它将显示有关系统中 GPU 的信息。这是我笔记本电脑上的一个示例：

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 426.00       Driver Version: 426.00       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce RTX 207... WDDM  | 00000000:01:00.0 Off |                  N/A |
    | N/A   38C    P8    12W /  N/A |    161MiB /  8192MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

以下是上述输出中的一些有用信息：

+ 从第 1 行我们可以看到 Nvidia 驱动程序版本 （426.00） 和 CUDA 版本 （10.1）。
+ 系统中只有一个 GPU，即 GeForce RTX 2070。它的设备 ID 为 0。如果 GPU 较多，则将从 0 开始标记。您可以在命令行中使用以下命令，在运行 CUDA 程序之前选择使用设备 1：

```
$ export CUDA_VISIBLE_DEVICES=1        
```

+ 此 GPU 处于 **WDDM（Windows 显示驱动程序模型）** 模式。另一种可能的模式是 **TCC（特斯拉计算集群），** 但它仅适用于 Tesla、Quadro 和 Titan 系列的 GPU。可以使用以下命令来选择模式（在 Windows 中，需要具有管理员权限，并且应删除下面的 `sudo`）：

```
$ sudo nvidia-smi -g GPU_ID -dm 0 # set device GPU_ID to the WDDM mode
$ sudo nvidia-smi -g GPU_ID -dm 1 # set device GPU_ID to the TCC mode
```

+ `Compute M`。指计算模式。这里的计算模式是`默认`的，这意味着允许使用 GPU 运行多个计算过程。另一种可能的模式是 `E. Process`，意思是独占进程模式。只能运行一个一个计算进程独占该GPU。`E. Process`模式不适用于 WDDM 模式下的 GPU。可以使用以下命令来选择模式（在 Windows 中，需要具有管理员权限，并且应删除下面的 `sudo`）：

```
$ sudo nvidia-smi -i GPU_ID -c 0 # set device GPU_ID to Default mode
$ sudo nvidia-smi -i GPU_ID -c 1 # set device GPU_ID to E. Process mode
```

有关 `nvidia-smi` 程序的更多详细信息，请参阅以下官方手册：https://developer.nvidia.com/nvidia-system-management-interface
