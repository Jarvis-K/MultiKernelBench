from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from dataset import dataset

# AscendC算子API的完整参考
ASCENDC_API_REFERENCE = {
    "基础运算": {
        "Add": "AscendC::Add(output, input1, input2, length)",
        "Sub": "AscendC::Sub(output, input1, input2, length)", 
        "Mul": "AscendC::Mul(output, input1, input2, length)",
        "Div": "AscendC::Div(output, input1, input2, length)"
    },
    "激活函数": {
        "Relu": "AscendC::Relu(output, input, length)",
        "LeakyRelu": "AscendC::LeakyRelu(output, input, negative_slope, length)",
        "Sigmoid": "AscendC::Sigmoid(output, input, length)",
        "Tanh": "AscendC::Tanh(output, input, length)",
        "Gelu": "AscendC::Gelu(output, input, length)"
    },
    "归一化": {
        "LayerNorm": "AscendC::LayerNorm(output, input, mean, var, weight, bias, eps, length)",
        "BatchNorm": "AscendC::BatchNorm(output, input, running_mean, running_var, weight, bias, momentum, eps, length)"
    },
    "卷积": {
        "Conv2d": "AscendC::Conv2d(output, input, weight, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, length)"
    },
    "矩阵运算": {
        "Matmul": "AscendC::Matmul(output, input1, input2, m, k, n, alpha, beta, length)"
    }
}

def get_ascendc_api_guide():
    """生成AscendC API参考指南"""
    guide = """
## AscendC算子API参考指南

### 基础运算
"""
    for category, apis in ASCENDC_API_REFERENCE.items():
        guide += f"\n#### {category}\n"
        for api_name, api_signature in apis.items():
            guide += f"- **{api_name}**: `{api_signature}`\n"
    
    guide += """
### 内存管理API
- `AscendC::LocalTensor<T>`: 本地张量类型
- `AscendC::GlobalTensor<T>`: 全局张量类型
- `AscendC::DataCopy(dst, src, length)`: 数据拷贝
- `AscendC::GetBlockNum()`: 获取块数量
- `AscendC::GetBlockIdx()`: 获取当前块索引

### 性能优化建议
1. 合理设置BUFFER_NUM: 通常设置为2-4
2. 优化tile大小: 根据数据大小和内存限制调整
3. 减少内存拷贝: 尽可能复用Local Memory
4. 利用向量化: 使用AICore的向量化能力
5. 流水线并行: 使用CopyIn-Compute-CopyOut模式
"""
    
    return guide

@register_prompt("ascendc", "agent")
class AscendcAgentPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 获取算子类别
        category = dataset[op]['category']
        
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加AscendC API参考指南
        api_guide = get_ascendc_api_guide()
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + api_guide
        
        return full_prompt
