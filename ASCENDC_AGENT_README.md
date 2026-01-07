# AscendC算子编写Agent

基于AscendC算子API文档构建的智能算子编写Agent，为不同类型的算子提供专门的prompt生成器。

## 功能特性

### 1. 算子类型专用Prompt生成器

- **激活函数算子** (`ascendc_activation_specific.py`)
  - 支持ReLU、LeakyReLU、Sigmoid、Tanh、GELU、Swish等激活函数
  - 提供数值稳定性指导
  - 包含内存优化建议

- **归一化算子** (`ascendc_normalization_specific.py`)
  - 支持LayerNorm、BatchNorm、GroupNorm、InstanceNorm
  - 提供统计信息计算指导
  - 包含参数管理和内存优化建议

- **卷积算子** (`ascendc_convolution_specific.py`)
  - 支持标准卷积、转置卷积、深度卷积
  - 提供参数设置和tiling策略指导
  - 包含性能优化建议

- **矩阵乘法算子** (`ascendc_matmul_specific.py`)
  - 支持标准矩阵乘法和批量矩阵乘法
  - 提供维度处理和内存布局指导
  - 包含数值稳定性建议

- **通用算子** (`ascendc_operator_specific.py`)
  - 覆盖所有算子类型的通用指导
  - 提供AscendC API参考
  - 包含性能优化建议

### 2. 综合Prompt生成器 (`ascendc_comprehensive.py`)

- **完整的编程约束**：包含全局约束、编程范式、对齐要求等
- **通用代码骨架**：提供标准的CopyIn-Compute-CopyOut三段式模板
- **算子特定指导**：根据算子类型提供专门的实现模板
- **实用代码片段**：包含常用的代码片段和最佳实践

### 3. 智能Agent (`ascendc_agent.py`)

- 自动识别算子类型并选择合适的prompt策略
- 提供完整的AscendC API参考指南
- 集成所有算子类型的指导信息

## 使用方法

### 1. 使用综合Prompt生成器（推荐）

```bash
# 使用综合prompt生成器，包含完整的编程指导和约束
python generate_and_write.py --model deepseek-chat --language ascendc --strategy comprehensive --categories activation
```

### 2. 使用特定算子类型的Prompt生成器

```bash
# 激活函数算子
python generate_and_write.py --model deepseek-chat --language ascendc --strategy activation_specific --categories activation

# 归一化算子
python generate_and_write.py --model deepseek-chat --language ascendc --strategy normalization_specific --categories normalization

# 卷积算子
python generate_and_write.py --model deepseek-chat --language ascendc --strategy convolution_specific --categories convolution

# 矩阵乘法算子
python generate_and_write.py --model deepseek-chat --language ascendc --strategy matmul_specific --categories matmul
```

### 3. 使用通用Agent

```bash
# 使用智能Agent自动选择合适的策略
python generate_and_write.py --model deepseek-chat --language ascendc --strategy agent --categories activation normalization convolution matmul
```

### 4. 使用通用算子指导

```bash
# 使用通用算子指导
python generate_and_write.py --model deepseek-chat --language ascendc --strategy operator_specific --categories all
```

## 支持的算子类型

### 激活函数 (activation)
- `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `swish`
- `elu`, `softplus`, `softsign`, `hardsigmoid`, `hardtanh`
- `selu`, `min_gpt_new_gelu`, `log_softmax`, `softmax`

### 归一化 (normalization)
- `layer_norm`, `batch_norm`, `group_norm`, `instance_norm`

### 卷积 (convolution)
- `conv_standard_2d`, `conv_standard_3d`
- `conv_transposed_2d`, `conv_transposed_3d`
- `conv_depthwise_2d`, `conv_depthwise_separable_2d`

### 矩阵乘法 (matmul)
- `matmul`, `batched_matmul`
- `four_dim_tensor_matrix_multiplication`

### 其他类型
- **算术运算** (arithmetic): `add`, `sub`, `mul`, `div`
- **广播** (broadcast): 各种广播操作
- **归约** (reduce): `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`
- **池化** (pooling): `max_pool`, `avg_pool`
- **损失函数** (loss): `cross_entropy`, `mse_loss`
- **融合算子** (fuse): 各种算子融合

## 编程约束和最佳实践

### 全局约束
- **仅使用AscendC API**：LocalTensor, TQue/TQueBind, TPipe, AllocTensor/FreeTensor, EnQue/DeQue, DataCopy
- **三阶段模式**：CopyIn → Compute → CopyOut
- **内存管理**：GM↔片上存储使用DataCopy，所有中间张量使用LocalTensor和队列管理
- **同步保证**：必要处插入同步（barrier/队列入出保证顺序）

### 对齐与边界
- GM地址与长度符合32B对齐要求
- 非整除的尾块使用掩码或分段处理
- 避免越界读写

### 数值稳定性
- Softmax/LogSoftmax使用rowwise max + subtract + exp + sum的稳定实现
- 归一化使用eps防护
- 优化器避免除零

### 可编译要求
- 给出`extern "C" __global__ __aicore__ void kernel(...)`形式的核函数
- 参数使用`__gm__`指针与标量
- 不引入除AscendC以外的第三方库
- 包含必要的头文件与命名空间声明

## AscendC API参考

### 基础运算API
```cpp
// 算术运算
AscendC::Add(output, input1, input2, length);
AscendC::Sub(output, input1, input2, length);
AscendC::Mul(output, input1, input2, length);
AscendC::Div(output, input1, input2, length);

// 激活函数
AscendC::Relu(output, input, length);
AscendC::LeakyRelu(output, input, negative_slope, length);
AscendC::Sigmoid(output, input, length);
AscendC::Tanh(output, input, length);
AscendC::Gelu(output, input, length);
```

### 高级算子API
```cpp
// 归一化
AscendC::LayerNorm(output, input, mean, var, weight, bias, eps, length);
AscendC::BatchNorm(output, input, running_mean, running_var, weight, bias, momentum, eps, length);

// 卷积
AscendC::Conv2d(output, input, weight, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, length);

// 矩阵乘法
AscendC::Matmul(output, input1, input2, m, k, n, alpha, beta, length);
```

### 内存管理API
```cpp
// 张量类型
AscendC::LocalTensor<T> localTensor;
AscendC::GlobalTensor<T> globalTensor;

// 数据操作
AscendC::DataCopy(dst, src, length);
AscendC::GetBlockNum();
AscendC::GetBlockIdx();
```

## 性能优化建议

### 1. 内存管理
- 合理设置`BUFFER_NUM`（通常为2-4）
- 优化tile大小以平衡计算和内存
- 减少不必要的内存拷贝
- 复用Local Memory

### 2. 计算优化
- 利用AICore的向量化能力
- 使用流水线并行（CopyIn-Compute-CopyOut）
- 优化数据访问模式
- 合理设置block和tile维度

### 3. 数值稳定性
- 注意float16、float32等不同精度
- 避免数值溢出和下溢
- 使用合适的eps值
- 验证边界条件

## 文件结构

```
prompt_generators/
├── ascendc_activation_specific.py    # 激活函数专用
├── ascendc_normalization_specific.py # 归一化专用
├── ascendc_convolution_specific.py   # 卷积专用
├── ascendc_matmul_specific.py        # 矩阵乘法专用
├── ascendc_operator_specific.py      # 通用算子指导
├── ascendc_comprehensive.py          # 综合prompt生成器
├── ascendc_agent.py                  # 智能Agent
└── prompt_registry.py                # Prompt注册器
```

## 扩展指南

### 添加新的算子类型

1. 创建新的prompt生成器文件
2. 继承`BasePromptStrategy`类
3. 使用`@register_prompt`装饰器注册
4. 实现`generate`方法

### 添加新的API函数

1. 在相应的prompt生成器中添加API函数信息
2. 更新实现示例
3. 添加参数说明和注意事项

## 注意事项

1. **命名规范**: 确保kernel函数名和PascalCase命名正确
2. **数据类型**: 注意处理不同的数据类型和精度
3. **内存管理**: 正确管理Local Memory的分配和释放
4. **性能测试**: 对生成的算子进行性能基准测试
5. **数值验证**: 确保数值精度和稳定性

## 参考文档

- [AscendC算子API文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0003.html)
- [MultiKernelBench项目文档](README.md)
