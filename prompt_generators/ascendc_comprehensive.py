from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from dataset import dataset

# 全局约束
GLOBAL_CONSTRAINTS = """
## 全局约束（必须遵守）

### 编程范式
- 仅使用AscendC API：LocalTensor, TQue/TQueBind, TPipe, AllocTensor/FreeTensor, EnQue/DeQue, DataCopy
- 三阶段模式：CopyIn → Compute → CopyOut
- 内存管理：GM↔片上存储使用DataCopy，所有中间张量使用LocalTensor和队列管理
- 同步保证：必要处插入同步（barrier/队列入出保证顺序）

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
"""

# 通用代码骨架
COMMON_SKELETON = """
## 通用代码骨架

```cpp
#include <ascendc/ascendc.h>
using namespace AscendC;

extern "C" __global__ __aicore__
void KERNEL_NAME(/* 形参 */) {
  // 1) 定义队列与管道
  TPipe pipe;
  TQueBind<TPosition::VECIN,  TPosition::GM, 1>  qIn;
  TQueBind<TPosition::VECOUT, TPosition::GM, 1>  qOut;
  TQueBind<TPosition::VECCALC, TPosition::VECCALC, 1> qTmp;

  // 2) 初始化缓冲
  int64_t num = /* tile块数 */;  
  int64_t len = /* 每块长度(元素) */;
  pipe.InitBuffer(qIn,  num, len);
  pipe.InitBuffer(qOut, num, len);

  // 3) CopyIn：GM→Local
  {
    LocalTensor<DTYPE> xLocal = qIn.AllocTensor<DTYPE>();
    DataCopy(xLocal, xGlobal, len);
    qIn.EnQue(xLocal);
  }

  // 4) Compute：DeQue→计算→EnQue
  {
    LocalTensor<DTYPE> x = qIn.DeQue<DTYPE>();
    LocalTensor<DTYPE> y = qOut.AllocTensor<DTYPE>();
    // === 计算逻辑 ===
    qOut.EnQue(y);
    qIn.FreeTensor(x);
  }

  // 5) CopyOut：Local→GM
  {
    LocalTensor<DTYPE> yLocal = qOut.DeQue<DTYPE>();
    DataCopy(yGlobal, yLocal, len);
    qOut.FreeTensor(yLocal);
  }
}
```
"""

# 算子类型指导
OPERATOR_GUIDES = {
    "activation": {
        "description": "激活函数：对输入逐元素变换",
        "template": """
```cpp
// 以LeakyReLU为例
LocalTensor<DTYPE> x = qIn.DeQue<DTYPE>();
LocalTensor<DTYPE> y = qOut.AllocTensor<DTYPE>();
// y = x >= 0 ? x : alpha * x
Select(y, x, /*cond: x>=0*/, /*true*/x, /*false*/(alpha * x), len);
qOut.EnQue(y); 
qIn.FreeTensor(x);
```
""",
    },
    "normalization": {
        "description": "归一化：计算统计信息并应用仿射变换",
        "template": """
```cpp
LocalTensor<T> x = qIn.DeQue<T>();
LocalTensor<T> y = qOut.AllocTensor<T>();
LocalTensor<T> mean = qTmp.AllocTensor<T>();
LocalTensor<T> var  = qTmp.AllocTensor<T>();

ReduceMean(mean, x, /*axisSet=*/AXIS);
VectorSquare(var, x, len);
ReduceMean(var, var, AXIS);
VectorFma(var, mean, -mean, /*E[x^2]-mean^2*/);
VectorAddScalar(var, var, eps);
VectorRsqrt(var, var, len);
NormalizeAffine(y, x, mean, var, gamma, beta, len);
qOut.EnQue(y);
```
""",
    },
    "convolution": {
        "description": "卷积：滑动窗口特征提取",
        "template": """
```cpp
// MatMul C = A*B，按K分块累加
for (int k0 = 0; k0 < K; k0 += TK) {
  LocalTensor<T> a = qIn.DeQue<T>();
  LocalTensor<T> b = qIn2.DeQue<T>();
  MMAD(acc, a, b, /*tile params*/);
}
Store(acc, Ctile);
```
""",
    }
}

def get_comprehensive_prompt(op_name):
    """生成综合的AscendC prompt"""
    category = dataset[op_name]['category']
    op_guide = OPERATOR_GUIDES.get(category, OPERATOR_GUIDES["activation"])
    
    prompt = f"""
# AscendC算子编写指南 - {op_name.upper()}

{GLOBAL_CONSTRAINTS}

{COMMON_SKELETON}

## 算子特定指导 - {category.upper()}

### 算子信息
- **算子名称**: {op_name}
- **类别**: {category}
- **描述**: {op_guide['description']}

### 实现模板
{op_guide['template']}

## 输出要求
- 完整AscendC kernel（含`__global__ __aicore__`）
- 参数定义（使用`__gm__`指针）
- CopyIn/Compute/CopyOut三段式实现
- 必要的边界与数值稳定处理
- 注释说明关键API使用
"""
    
    return prompt

@register_prompt("ascendc", "comprehensive")
class AscendcComprehensivePromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加综合指导
        comprehensive_guide = get_comprehensive_prompt(op)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + comprehensive_guide
        
        return full_prompt
