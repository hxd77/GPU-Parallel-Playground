# 🚀 CUDA 并行计算项目



本仓库用于学习与实践 **NVIDIA CUDA 并行编程**，通过示例代码与实验验证 GPU 加速的高性能计算能力。  
内容涵盖从入门到进阶的 CUDA 编程技巧，包括线程组织、内存优化、核函数设计、性能分析等。



---

<p align="center">
    <a href="https://github.com/hxd77/GPU-Parallel-Playground"><img src="https://raw.githubusercontent.com/hxd77/BlogImage/master/TyporaImage/20251016222120222.png"</a>
</p>

 <p align="center">
<img src="https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green" alt="Coverage">
    <img src="https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white">
     <img src="https://img.shields.io/badge/VS%20Code%20Insiders-35b393.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white">
    <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Package version">
    </p>

> 本书主要参考了[CUDA编程基础与实战]([CUDA编程_百度百科](https://baike.baidu.com/item/CUDA编程/59848340))，书籍[Github地址](https://github.com/brucefan1983/CUDA-Programming)

## 📘 项目简介

CUDA（Compute Unified Device Architecture）是 NVIDIA 提供的并行计算平台和编程模型。  
通过 CUDA，开发者可以利用 GPU 的并行处理能力实现对数据密集型任务的加速。

本项目包含以下内容：

- 🧮 **基础示例**：CUDA 向量加法、矩阵乘法、归约求和等  
- ⚙️ **内存优化**：共享内存、常量内存、寄存器使用  
- 🔄 **线程模型**：线程块 (Block) 与线程网格 (Grid) 的设计  
- 📊 **性能分析**：使用 `nvprof`、`nsight` 工具进行性能分析  
- 🧠 **进阶内容（可选）**：GPU 并行算法、Warp 同步、动态并行、流（Stream）机制  

---

## 📂 项目结构

```

├── src/ # CUDA 源码 (.cu)  
│ ├── vector_add.cu # 向量加法示例  
│ ├── matrix_mul.cu # 矩阵乘法示例  
│ ├── reduce_sum.cu # 归约求和示例  
│ └── ...  
├── include/ # 头文件  
├── data/ # 测试数据（可选）  
├── docs/ # 学习笔记 / 文档  
├── Makefile # 编译脚本  
└── README.md

```

---

## ⚡ 编译与运行

### 1️⃣ 环境要求
- NVIDIA GPU（支持 CUDA Compute Capability 3.0 以上）
- CUDA Toolkit（推荐 11.x 或更高版本）
- GCC 编译器（建议 9.0+）
- Linux / Windows / WSL2 环境均可

### 2️⃣ 编译
```bash
nvcc src/vector_add.cu -o vector_add
```

或使用 `Makefile`：

```bash
make all
```

### 3️⃣ 运行

```bash
./vector_add
```

* * *

## 🧩 示例输出

```text
Vector addition using GPU
Array size: 1024
Execution time: 0.034 ms
Result verified: SUCCESS ✅
```

* * *

## 📈 性能对比

| 任务               | CPU耗时(ms) | GPU耗时(ms) | 加速比 |
| ------------------ | ----------- | ----------- | ------ |
| 向量加法 (1e6)     | 5.12        | 0.04        | 128x   |
| 矩阵乘法 (512x512) | 382.3       | 12.7        | 30x    |

* * *

## 🧭 学习参考

* CUDA 官方文档
  
* NVIDIA 开发者社区
  
* 《CUDA C Programming Guide》
  
* Parallel Forall 博客
  

* * *

## 💡 TODO

*  添加更多示例（如卷积、归约、前缀和）
  
*  增加性能优化对比分析
  
*  整理 GPU 架构与线程调度笔记
  
*  支持多 GPU 并行计算
  

* * *

## 📜 许可证

本项目基于 **MIT License** 开源，欢迎 fork 与学习。

* * *

## ❤️ 致谢

感谢 NVIDIA 提供强大的 CUDA 平台，也感谢所有为开源 GPU 生态做出贡献的开发者。

