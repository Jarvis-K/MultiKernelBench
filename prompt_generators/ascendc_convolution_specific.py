from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template

# 卷积算子的具体实现指导
CONVOLUTION_OPERATOR_GUIDES = {
    "conv_standard_2d": {
        "api_function": "AscendC::Conv2d",
        "description": "标准2D卷积：使用滑动窗口进行特征提取",
        "implementation": """
        // Conv2d实现示例
        AscendC::LocalTensor<DTYPE_X> inputLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> weightLocal = weightQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> outputLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        
        AscendC::Conv2d(outputLocal, inputLocal, weightLocal, 
                       this->kernel_h, this->kernel_w, this->stride_h, this->stride_w, 
                       this->pad_h, this->pad_w, this->dilation_h, this->dilation_w, 
                       this->tileLength);
        """,
        "parameters": "需要kernel_size、stride、padding、dilation参数"
    }
}

def get_convolution_specific_prompt(op_name):
    """为卷积算子生成特定的prompt"""
    op_guide = CONVOLUTION_OPERATOR_GUIDES.get(op_name, CONVOLUTION_OPERATOR_GUIDES["conv_standard_2d"])
    
    prompt = f"""
## AscendC卷积算子编写指南

### 算子信息
- **算子名称**: {op_name}
- **API函数**: {op_guide['api_function']}
- **描述**: {op_guide['description']}
- **参数要求**: {op_guide['parameters']}

### 核心实现
```cpp
{op_guide['implementation']}
```

### 卷积算子的特殊注意事项

#### 1. 参数设置
- kernel_size: 卷积核大小，影响感受野
- stride: 步长，影响输出特征图尺寸
- padding: 填充，保持特征图尺寸
- dilation: 膨胀，增加感受野

#### 2. 内存管理
- 输入特征图需要从Global Memory加载
- 卷积权重通常较小，可以预加载
- 输出特征图计算结果写回Global Memory

#### 3. 计算优化
- 利用AICore的并行能力
- 优化数据访问模式
- 合理设置tile大小

### 测试建议
- 测试不同kernel_size、stride、padding组合
- 验证输出形状计算的正确性
- 检查数值精度和稳定性
- 性能基准测试
"""
    
    return prompt

@register_prompt("ascendc", "convolution_specific")
class AscendcConvolutionSpecificPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加卷积算子特定的指导
        convolution_guide = get_convolution_specific_prompt(op)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + convolution_guide
        
        return full_prompt
