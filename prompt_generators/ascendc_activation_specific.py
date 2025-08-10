from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template

# 激活函数算子的具体实现指导
ACTIVATION_OPERATOR_GUIDES = {
    "relu": {
        "api_function": "AscendC::Relu",
        "description": "ReLU激活函数：f(x) = max(0, x)",
        "implementation": """
        // ReLU实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Relu(outputLocal, inputLocal, this->tileLength);
        """,
        "parameters": "无需额外参数"
    },
    "leaky_relu": {
        "api_function": "AscendC::LeakyRelu", 
        "description": "LeakyReLU激活函数：f(x) = max(αx, x)，其中α为负斜率",
        "implementation": """
        // LeakyReLU实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        float negative_slope = 0.01f; // 从tiling数据或参数获取
        AscendC::LeakyRelu(outputLocal, inputLocal, negative_slope, this->tileLength);
        """,
        "parameters": "需要negative_slope参数"
    },
    "sigmoid": {
        "api_function": "AscendC::Sigmoid",
        "description": "Sigmoid激活函数：f(x) = 1 / (1 + e^(-x))",
        "implementation": """
        // Sigmoid实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Sigmoid(outputLocal, inputLocal, this->tileLength);
        """,
        "parameters": "无需额外参数"
    },
    "tanh": {
        "api_function": "AscendC::Tanh",
        "description": "Tanh激活函数：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
        "implementation": """
        // Tanh实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Tanh(outputLocal, inputLocal, this->tileLength);
        """,
        "parameters": "无需额外参数"
    },
    "gelu": {
        "api_function": "AscendC::Gelu",
        "description": "GELU激活函数：f(x) = x * Φ(x)，其中Φ是标准正态分布的累积分布函数",
        "implementation": """
        // GELU实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Gelu(outputLocal, inputLocal, this->tileLength);
        """,
        "parameters": "无需额外参数"
    },
    "swish": {
        "api_function": "AscendC::Swish",
        "description": "Swish激活函数：f(x) = x * sigmoid(x)",
        "implementation": """
        // Swish实现示例（需要组合Sigmoid和Mul）
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_X> sigmoidOutput = tempQueue.AllocTensor<DTYPE_X>();
        
        // 计算sigmoid
        AscendC::Sigmoid(sigmoidOutput, inputLocal, this->tileLength);
        // 计算x * sigmoid(x)
        AscendC::Mul(outputLocal, inputLocal, sigmoidOutput, this->tileLength);
        
        tempQueue.FreeTensor(sigmoidOutput);
        """,
        "parameters": "需要临时缓冲区存储sigmoid结果"
    }
}

def get_activation_specific_prompt(op_name):
    """为激活函数算子生成特定的prompt"""
    op_guide = ACTIVATION_OPERATOR_GUIDES.get(op_name, ACTIVATION_OPERATOR_GUIDES["relu"])
    
    prompt = f"""
## AscendC激活函数算子编写指南

### 算子信息
- **算子名称**: {op_name}
- **API函数**: {op_guide['api_function']}
- **描述**: {op_guide['description']}
- **参数要求**: {op_guide['parameters']}

### 核心实现
```cpp
{op_guide['implementation']}
```

### 激活函数算子的特殊注意事项

#### 1. 数据类型处理
- 激活函数通常保持输入输出数据类型一致
- 注意处理float16、float32等不同精度
- 某些激活函数可能需要额外的精度控制

#### 2. 数值稳定性
- 对于Sigmoid、Tanh等函数，注意数值溢出问题
- 使用合适的数值范围限制
- 考虑使用更稳定的实现方式

#### 3. 性能优化
- 激活函数计算相对简单，重点优化内存访问模式
- 利用AICore的向量化能力
- 合理设置tile大小以充分利用计算资源

#### 4. 内存管理
- 激活函数通常只需要一个输入和一个输出
- 可以复用输入tensor作为输出（原地操作）
- 注意Local Memory的使用效率

#### 5. 特殊激活函数实现
对于复杂的激活函数（如Swish、GELU），可能需要：
- 组合多个基础算子
- 使用临时缓冲区存储中间结果
- 实现自定义的数值计算

### 代码模板结构
```cpp
class Kernel{underscore_to_pascalcase(op_name)} {{
public:
    __aicore__ inline void Compute(int32_t progress) {{
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        
        // 在这里实现具体的激活函数逻辑
        {op_guide['api_function']}(outputLocal, inputLocal, this->tileLength);
        
        outQueueZ.EnQue<DTYPE_Z>(outputLocal);
        inQueueX.FreeTensor(inputLocal);
    }}
}};
```

### 测试建议
- 测试不同数值范围的输入
- 验证边界条件（如0、极大值、极小值）
- 检查数值精度和稳定性
- 性能基准测试
"""
    
    return prompt

@register_prompt("ascendc", "activation_specific")
class AscendcActivationSpecificPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加激活函数特定的指导
        activation_guide = get_activation_specific_prompt(op)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + activation_guide
        
        return full_prompt
