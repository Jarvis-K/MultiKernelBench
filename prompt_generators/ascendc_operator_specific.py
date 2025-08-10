from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template

# AscendC算子API文档中的主要算子类型
ASCENDC_OPERATOR_TYPES = {
    "activation": {
        "description": "激活函数算子，包括ReLU、LeakyReLU、Sigmoid、Tanh、GELU等",
        "api_functions": ["AscendC::Relu", "AscendC::LeakyRelu", "AscendC::Sigmoid", "AscendC::Tanh", "AscendC::Gelu"],
        "template_hints": [
            "使用AscendC::Relu进行ReLU激活",
            "使用AscendC::LeakyRelu进行LeakyReLU激活，需要设置negative_slope参数",
            "使用AscendC::Sigmoid进行Sigmoid激活",
            "使用AscendC::Tanh进行Tanh激活",
            "使用AscendC::Gelu进行GELU激活"
        ]
    },
    "arithmetic": {
        "description": "算术运算算子，包括Add、Sub、Mul、Div等",
        "api_functions": ["AscendC::Add", "AscendC::Sub", "AscendC::Mul", "AscendC::Div"],
        "template_hints": [
            "使用AscendC::Add进行张量加法运算",
            "使用AscendC::Sub进行张量减法运算",
            "使用AscendC::Mul进行张量乘法运算",
            "使用AscendC::Div进行张量除法运算"
        ]
    },
    "normalization": {
        "description": "归一化算子，包括LayerNorm、BatchNorm、GroupNorm等",
        "api_functions": ["AscendC::LayerNorm", "AscendC::BatchNorm", "AscendC::GroupNorm"],
        "template_hints": [
            "使用AscendC::LayerNorm进行层归一化，需要计算均值和方差",
            "使用AscendC::BatchNorm进行批归一化，需要统计批次统计信息",
            "使用AscendC::GroupNorm进行组归一化，按组计算统计信息"
        ]
    },
    "convolution": {
        "description": "卷积算子，包括2D/3D卷积、转置卷积等",
        "api_functions": ["AscendC::Conv2d", "AscendC::Conv3d", "AscendC::ConvTranspose2d"],
        "template_hints": [
            "使用AscendC::Conv2d进行2D卷积运算，需要设置kernel_size、stride、padding等参数",
            "使用AscendC::Conv3d进行3D卷积运算",
            "使用AscendC::ConvTranspose2d进行2D转置卷积运算"
        ]
    },
    "matmul": {
        "description": "矩阵乘法算子",
        "api_functions": ["AscendC::Matmul"],
        "template_hints": [
            "使用AscendC::Matmul进行矩阵乘法运算，支持批量矩阵乘法",
            "注意矩阵维度的匹配和转置操作"
        ]
    },
    "pooling": {
        "description": "池化算子，包括MaxPool、AvgPool等",
        "api_functions": ["AscendC::MaxPool2d", "AscendC::AvgPool2d"],
        "template_hints": [
            "使用AscendC::MaxPool2d进行最大池化，需要设置kernel_size、stride、padding",
            "使用AscendC::AvgPool2d进行平均池化"
        ]
    },
    "reduce": {
        "description": "归约算子，包括Sum、Mean、Max、Min等",
        "api_functions": ["AscendC::ReduceSum", "AscendC::ReduceMean", "AscendC::ReduceMax", "AscendC::ReduceMin"],
        "template_hints": [
            "使用AscendC::ReduceSum进行求和归约，需要指定归约维度",
            "使用AscendC::ReduceMean进行平均归约",
            "使用AscendC::ReduceMax进行最大值归约",
            "使用AscendC::ReduceMin进行最小值归约"
        ]
    },
    "loss": {
        "description": "损失函数算子，包括CrossEntropy、MSE等",
        "api_functions": ["AscendC::CrossEntropy", "AscendC::MseLoss"],
        "template_hints": [
            "使用AscendC::CrossEntropy进行交叉熵损失计算",
            "使用AscendC::MseLoss进行均方误差损失计算"
        ]
    }
}

def get_operator_specific_prompt(op_name, category):
    """根据算子类型生成特定的prompt提示"""
    if category not in ASCENDC_OPERATOR_TYPES:
        return ""
    
    op_info = ASCENDC_OPERATOR_TYPES[category]
    prompt = f"""
## AscendC算子编写指南 - {category.upper()}类型

### 算子描述
{op_info['description']}

### 相关API函数
{', '.join(op_info['api_functions'])}

### 实现要点
"""
    
    for hint in op_info['template_hints']:
        prompt += f"- {hint}\n"
    
    prompt += f"""
### 针对{op_name}算子的特殊要求
1. 确保kernel函数名称为: {op_name}_custom
2. 在project_json_src中使用PascalCase命名: {underscore_to_pascalcase(op_name)}Custom
3. 根据算子特性选择合适的AscendC API函数
4. 正确处理数据类型和内存布局
5. 优化tiling策略以提高性能

### 内存管理注意事项
- 使用AscendC::TPipe进行流水线管理
- 合理设置BUFFER_NUM和tile大小
- 注意Global Memory和Local Memory的数据拷贝
- 正确释放Local Tensor资源

### 性能优化建议
- 利用AICore的向量化能力
- 合理设置block和tile维度
- 减少不必要的内存拷贝
- 使用合适的tiling策略
"""
    
    return prompt

@register_prompt("ascendc", "operator_specific")
class AscendcOperatorSpecificPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 获取算子的类别信息
        from dataset import dataset
        category = dataset[op]['category']
        
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加算子特定的指导
        operator_guide = get_operator_specific_prompt(op, category)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + operator_guide
        
        return full_prompt
