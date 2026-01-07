from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template

# 归一化算子的具体实现指导
NORMALIZATION_OPERATOR_GUIDES = {
    "layer_norm": {
        "api_function": "AscendC::LayerNorm",
        "description": "层归一化：对指定维度进行归一化，计算均值和方差",
        "implementation": """
        // LayerNorm实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_X> weightLocal = weightQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> biasLocal = biasQueue.DeQue<DTYPE_X>();
        
        // 计算均值和方差
        AscendC::LocalTensor<DTYPE_X> meanLocal = tempQueue.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> varLocal = tempQueue.AllocTensor<DTYPE_X>();
        
        AscendC::ReduceMean(meanLocal, inputLocal, this->normalized_shape, this->tileLength);
        AscendC::ReduceVar(varLocal, inputLocal, meanLocal, this->normalized_shape, this->tileLength);
        
        // 应用LayerNorm
        AscendC::LayerNorm(outputLocal, inputLocal, meanLocal, varLocal, 
                          weightLocal, biasLocal, this->eps, this->tileLength);
        
        tempQueue.FreeTensor(meanLocal);
        tempQueue.FreeTensor(varLocal);
        """,
        "parameters": "需要weight、bias、eps参数，以及normalized_shape信息"
    }
}

def get_normalization_specific_prompt(op_name):
    """为归一化算子生成特定的prompt"""
    op_guide = NORMALIZATION_OPERATOR_GUIDES.get(op_name, NORMALIZATION_OPERATOR_GUIDES["layer_norm"])
    
    prompt = f"""
## AscendC归一化算子编写指南

### 算子信息
- **算子名称**: {op_name}
- **API函数**: {op_guide['api_function']}
- **描述**: {op_guide['description']}
- **参数要求**: {op_guide['parameters']}

### 核心实现
```cpp
{op_guide['implementation']}
```

### 归一化算子的特殊注意事项

#### 1. 统计信息计算
- 使用ReduceMean计算指定维度的均值
- 使用ReduceVar计算方差
- 添加eps避免除零错误

#### 2. 参数管理
- 权重和偏置作为可学习参数
- 需要从Global Memory加载参数
- 合理设置eps等超参数

#### 3. 内存管理
- 为均值、方差等中间结果分配Local Memory
- 多队列管理输入、权重、偏置等
- 合理复用Local Memory

### 测试建议
- 测试不同输入形状和维度
- 验证数值精度和稳定性
- 检查梯度计算的正确性
- 性能基准测试
"""
    
    return prompt

@register_prompt("ascendc", "normalization_specific")
class AscendcNormalizationSpecificPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加归一化算子特定的指导
        normalization_guide = get_normalization_specific_prompt(op)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + normalization_guide
        
        return full_prompt
