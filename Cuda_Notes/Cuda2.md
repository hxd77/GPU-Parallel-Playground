# CUDA第2章：CUDA中的线程组织

我们从最简单的 CUDA 程序开始：从 GPU 打印 `Hello World` 字符串。

## 2.1 C++ 中的 `Hello World` 程序

**要掌握 CUDA C++，必须首先掌握 C++**，但我们仍然从最简单的 C++ 程序之一开始：将 `Hello World` 消息打印到控制台（屏幕）。

要开发一个简单的 C++ 程序，可以按照以下步骤作：

+ 使用文本编辑器（例如 `gedit`;您可以选择任何您喜欢的内容）编写源代码。
+ 使用编译器编译源代码以获取目标文件，然后使用链接器链接目标文件和一些标准目标文件以获取可执行文件。编译和链接过程通常通过单个命令完成，我们将其简单地称为编译过程。
+ 运行可执行文件。

让我们首先在名为 [`hello.cpp`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello.cpp) 的源文件中编写以下程序。

```c++
#include <stdio.h>

int main(void)
{
    printf("Hello World!\n");
    return 0;
}
```

在 Linux 中，我们可以使用以下命令来编译它：

```
$ g++ hello.cpp
```

这将在当前目录中生成一个名为 `a.out` 的可执行文件。可以通过键入

```
$ ./a.out
```

然后，可以在控制台（屏幕）中看到以下消息：

```
Hello World!
```

还可以指定可执行文件的名称，例如

```
$ g++ hello.cpp -o hello
```

这将在当前目录中生成一个名为 `hello` 的可执行文件。可以通过键入

```
$ ./hello
```

在使用 MSVC 编译器 `cl.exe` 的 Windows 中，可以在命令提示符下使用以下命令编译程序：

````
$ cl.exe hello.cpp
````

这将生成一个名为 `hello.exe` 的可执行文件。可以使用以下命令运行它

```
$ hello.exe
```



## 2.2 CUDA 中的 `Hello World` 程序

在回顾了 C++ 中的 Hello World 程序之后，我们准备讨论 CUDA 中的类似程序。

### 2.2.1 仅包含主机函数的 CUDA 程序

我们实际上已经编写了一个有效的 CUDA 程序。这是因为 CUDA 编译器驱动 `nvcc` 可以通过调用主机编译器（如 `g++` 或 cl.exe）来编译纯 `C++` 代码。CUDA 源文件的默认后缀是 `.cu`，因此我们将 `hello.cpp` 重命名为 [`hello1.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello1.cu) 并使用以下命令对其进行编译：

```
$ nvcc hello1.cu
```

运行可执行文件的输出与以前相同。我们将在本章的最后一节中详细讨论 `nvcc`。现在读者只需要知道 `nvcc` 可用于编译带有 `.cu` 后缀的 CUDA 源文件。

### 2.2.2 包含 CUDA 内核的 CUDA 程序

尽管文件 `hello1.cu` 是使用 `nvcc` 编译的，但该程序没有使用 GPU。我们现在介绍一个真正使用 GPU 的程序。

我们知道 GPU 是设备，它需要主机给它命令。因此，一个典型的简单 CUDA 程序具有以下形式：

```c++
int main(void)
{
    host code
　  calling CUDA kernel(s)
　  host code
    return 0;
}
```

`CUDA 内核 `（或简称`内核 `）是由主机调用并在设备中执行的函数。定义内核的规则有很多，但现在我们只需要知道它必须用限定符 `__global__` 和 `void` 来装饰。这里，`__global__` 表示函数是`内核 `，`void` 表示 **CUDA 内核不能返回值** 。在内核内部， **几乎**所有的 C++ 构造都是允许的。

按照上述要求，我们编写一个内核，将消息打印到控制台：

```c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}    
```

限定符的顺序，`__global__` 和 `void`，并不重要。也就是说，我们也可以将内核写成：

```c++
void __global__ hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

然后我们编写一个 main 函数并从主机调用内核：

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

文件 [`hello2.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello2.cu) 可以编译如下：

```
$ nvcc hello2.cu
```

运行可执行文件，我们将从控制台看到以下消息：

```
Hello World from the GPU!
```

我们注意到内核的调用方式如下：

```c++
    hello_from_gpu<<<1, 1>>>();
```

内核名称和 `（）` 之间必须有一个**执行配置** ，例如 `<<<1、1>>>。` 执行配置指定内核的线程数及其组织。内核的线程形成一个网格，该**网格**可以包含多个**块** 。每个块可以包含多个线程。网格内的块数称为**网格大小** 。网格中的所有块都具有相同的线程数，这个数字称为**块大小** 。因此，网格中的线程总数是网格大小和块大小的乘积。对于一个简单的执行配置 `<<<grid_size， block_size>>> 两个`整数 `grid_size` 和 `block_size`（我们很快就会看到更多通用的执行配置），第一个数字 `grid_size` 是网格大小，第二个数字 `block_size` 是区块大小。在我们的 `hello2.cu` 程序中，执行配置 `<<<1， 1>>>` 表示网格大小和块大小都是 1，并且只有 `1 * 1 = 1` 个线程用于内核。

+ C++ 库 `<stdio.h>` 中的 `printf（）` 函数（也可以写为 `<cstdio>`）可以直接在内核中使用。但是，不能在内核中使用 `<iostream>` 库中的函数。声明

```c++
    cudaDeviceSynchronize();
```

使用内核调用同步**主机和设备**后，确保在从内核返回到主机之前，已刷新 `printf` 函数的输出流。如果没有这样的同步，主机将不会等待内核执行完成，并且消息不会输出到控制台。这是因为调用输出函数时，输出流是先放在缓冲区的，而缓冲区不会自动刷新。只有程序遇到某种同步操作是缓冲区才会刷新。`cudaDeviceSynchronize（）` 是我们将在本书课程中学习的众多 CUDA 运行时 API 函数之一。这里对同步的需求反映了**内核启动的异步性质** ，但我们不会费心在第 11 章之前详细说明它。



## 2.3 CUDA 中的线程组织

### 2.3.1 使用多线程的 CUDA 内核

GPU 中有许多内核，如果需要，可以为内核分配许多线程。以下程序 [`hello3.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello3.cu) 内核使用了一个有 2 个块的网格，每个块有 4 个线程：

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

因此，内核中使用的线程总数为 `2 * 4 = 8`。内核中的代码以一种称为“单指令多线程”的方式执行，这意味着内核（或网格中）中的每个线程都执行相同的指令序列（我们将在第 10 章中详细讨论）。因此，运行此程序的可执行文件会将以下文本打印到控制台：

```
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
    Hello World from the GPU!
```

上面的每一行都对应一个线程。但读者可能会问：哪条线是由哪条线产生的？我们将在下面回答这个问题。



### 2.3.2 在 CUDA 内核中使用线程索引

内核中的每个线程都有一个唯一的标识或索引。因为我们在执行配置中使用了两个数字（网格大小和块大小），所以内核中的每个线程也应该用两个数字来标识。在内核中，网格大小和块大小分别存储在内置变量 `gridDim.x` 和 `blockDim.x` 中。线程可以通过以下内置变量进行标识：

+ `blockIdx.x`：此变量指定**网格内线程的块索引** ，可以取从 0 到 `gridDim.x - 1` 的值。
+ `threadIdx.x`：该变量指定**块内线程的线程索引** ，可以取值从 0 到 `blockDim.x - 1`。

考虑一个执行配置为 `<<<10000， 256>>>` 的内核，我们知道网格大小 `gridDim.x` 为 10000，区块大小 `blockDim.x` 为 256。因此，内核中线程的块索引 `blockIdx.x` 可以取 0 到 9999 之间的值，线程的线程索引 `threadIdx.x` 可以取 0 到 255 之间的值。

回到我们的 `hello3.cu` 程序，我们给内核分配了 8 个线程，每个线程打印一行文本，但我们不知道哪行来自哪个线程。现在我们知道内核中的每个线程都可以唯一标识，我们可以使用它来告诉我们哪一行来自哪个线程。为此，我们重写程序以获得一个新程序，如 [`hello4.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello4.cu) 所示：

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

运行此程序的可执行文件，有时我们会得到以下输出，

```
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.
    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
```

有时我们会得到以下输出，

    Hello World from block 0 and thread 0.
    Hello World from block 0 and thread 1.
    Hello World from block 0 and thread 2.
    Hello World from block 0 and thread 3.
    Hello World from block 1 and thread 0.
    Hello World from block 1 and thread 1.
    Hello World from block 1 and thread 2.
    Hello World from block 1 and thread 3.

也就是说，有时块 0 先完成指令，有时块 1 先完成指令。这反映了 CUDA 内核执行的一个非常重要的特征，即**网格中的每个块都是相互独立的** 。



### 2.3.3 泛化到多维网格和块

读者可能已经注意到，上面介绍的 4 个内置变量使用了 C++ 中的`结构体`或`类`语法。这是真的：

+ `blockIdx` 和 `threadIdx` 的类型为 `uint3`，在 `vector_types.h` 中定义为：

```c
    struct __device_builtin__ uint3
    {
        unsigned int x, y, z;
    };    
    typedef __device_builtin__ struct uint3 uint3;
```

因此，除了 `blockIdx.x` 之外，我们还有 `blockIdx.y` 和 `blockIdx.z`。同样，除了 `threadIdx.x` 之外，我们还有 `threadIdx.y` 和 `threadIdx.z`。

+ `gridDim` 和 `blockDim` 的类型为 `dim3`，类似于 `uint3`，并有一些即将引入的构造函数。因此，除了 `gridDim.x`，我们还有 `gridDim.y` 和 `gridDim.z`。同样，除了 `blockDim.x` 之外，我们还有 `blockDim.y` 和 `blockDim.z`。
+ 因此，这些内置变量在三个维度上表示索引或大小：`x`、`y` 和 `z`。 **所有这些内置变量仅在 CUDA 内核中可见。**

我们可以使用 struct `dim3` 的构造函数来定义多维网格和块：

```c++
    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);
```

为了演示多维块的用法，我们编写了 Hello World 程序的最后一个版本 [`hello5.cu`](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/02-thread-organization/hello5.cu)：

```c++
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}

int main(void)
{
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();	//blockDim.x=2，blockDim.y=4，blockDim.z=1.gridDim.x=1
    cudaDeviceSynchronize();
    return 0;
}
```

该程序的输出为：

```
    Hello World from block-0 and thread-(0, 0)!
    Hello World from block-0 and thread-(1, 0)!
    Hello World from block-0 and thread-(0, 1)!
    Hello World from block-0 and thread-(1, 1)!
    Hello World from block-0 and thread-(0, 2)!
    Hello World from block-0 and thread-(1, 2)!
    Hello World from block-0 and thread-(0, 3)!
    Hello World from block-0 and thread-(1, 3)!
```

读者可能会注意到此处 `threadIdx.x` 和 `threadIdx.y` 的明确定义顺序。如果我们将线条从上到下标记为 0-7，则该标签可以计算为 `threadIdx.y * blockDim.x + threadIdx.x = threadIdx.y * 2 + threadIdx.x` ：

    Hello World from block-0 and thread-(0, 0)! // 0 = 0 * 2 + 0
    Hello World from block-0 and thread-(1, 0)! // 1 = 0 * 2 + 1
    Hello World from block-0 and thread-(0, 1)! // 2 = 1 * 2 + 0
    Hello World from block-0 and thread-(1, 1)! // 3 = 1 * 2 + 1
    Hello World from block-0 and thread-(0, 2)! // 4 = 2 * 2 + 0
    Hello World from block-0 and thread-(1, 2)! // 5 = 2 * 2 + 1
    Hello World from block-0 and thread-(0, 3)! // 6 = 3 * 2 + 0
    Hello World from block-0 and thread-(1, 3)! // 7 = 3 * 2 + 1
    