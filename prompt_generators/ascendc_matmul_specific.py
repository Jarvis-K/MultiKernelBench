from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from utils.utils import underscore_to_pascalcase
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template

# 矩阵乘法算子的具体实现指导
MATMUL_OPERATOR_GUIDES = {
    "matmul": {
        "api_function": "AscendC::Matmul",
        "description": "矩阵乘法：C = A @ B，支持批量矩阵乘法",
        "implementation": """
        // Matmul实现示例
        AscendC::LocalTensor<DTYPE_X> aLocal = inQueueA.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> bLocal = inQueueB.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> cLocal = outQueueC.AllocTensor<DTYPE_Z>();
        
        // 矩阵乘法 C = A @ B
        AscendC::Matmul(cLocal, aLocal, bLocal, 
                       this->m, this->k, this->n, 
                       this->alpha, this->beta, this->tileLength);
        """,
        "parameters": "需要矩阵维度m、k、n，以及alpha、beta参数"
    },
    "batched_matmul": {
        "api_function": "AscendC::BatchedMatmul",
        "description": "批量矩阵乘法：支持多个矩阵对的并行计算",
        "implementation": """
        // BatchedMatmul实现示例
        AscendC::LocalTensor<DTYPE_X> aLocal = inQueueA.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> bLocal = inQueueB.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> cLocal = outQueueC.AllocTensor<DTYPE_Z>();
        
        // 批量矩阵乘法
        AscendC::BatchedMatmul(cLocal, aLocal, bLocal,
                              this->batch_size, this->m, this->k, this->n,
                              this->alpha, this->beta, this->tileLength);
        """,
        "parameters": "需要batch_size、矩阵维度m、k、n，以及alpha、beta参数"
    }
}

def get_matmul_specific_prompt(op_name):
    """为矩阵乘法算子生成特定的prompt"""
    op_guide = MATMUL_OPERATOR_GUIDES.get(op_name, MATMUL_OPERATOR_GUIDES["matmul"])
    
    prompt = f"""
## AscendC矩阵乘法算子编写指南

### 算子信息
- **算子名称**: {op_name}
- **API函数**: {op_guide['api_function']}
- **描述**: {op_guide['description']}
- **参数要求**: {op_guide['parameters']}

### 核心实现
```cpp
{op_guide['implementation']}
```

### 矩阵乘法算子的特殊注意事项

#### 1. 维度处理
- **矩阵A**: 形状为 (m, k) 或 (batch_size, m, k)
- **矩阵B**: 形状为 (k, n) 或 (batch_size, k, n)
- **矩阵C**: 形状为 (m, n) 或 (batch_size, m, n)
- 确保维度匹配：A的列数 = B的行数

#### 2. 内存管理
- **输入矩阵**: 从Global Memory加载到Local Memory
- **输出矩阵**: 计算结果写回Global Memory
- **内存布局**: 注意矩阵的存储顺序（行主序/列主序）
- **tiling策略**: 合理分块以优化内存访问

#### 3. 计算优化
- **并行计算**: 利用AICore的并行能力
- **数据重用**: 优化数据访问模式，减少内存访问
- **缓存友好**: 合理设置tile大小以充分利用缓存

#### 4. 数值稳定性
- **精度控制**: 注意float16、float32等不同精度
- **数值范围**: 避免数值溢出和下溢
- **alpha/beta参数**: 正确设置缩放因子

#### 5. 特殊矩阵乘法
- **批量矩阵乘法**: 支持多个矩阵对的并行计算
- **转置操作**: 支持矩阵转置的矩阵乘法
- **稀疏矩阵**: 考虑稀疏矩阵的优化

### Tiling数据结构示例
```cpp
BEGIN_TILING_DATA_DEF(MatmulTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(float, alpha);
  TILING_DATA_FIELD_DEF(float, beta);
END_TILING_DATA_DEF;
```

### 测试建议
- 测试不同矩阵维度组合
- 验证数值精度和稳定性
- 检查批量矩阵乘法的正确性
- 性能基准测试
- 边界条件测试（如零矩阵、单位矩阵）
"""
    
    return prompt

@register_prompt("ascendc", "matmul_specific")
class AscendcMatmulSpecificPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        # 读取相关文件
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, 'add')
        
        # 生成基础模板
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, 'add')
        
        # 添加矩阵乘法算子特定的指导
        matmul_guide = get_matmul_specific_prompt(op)
        
        # 组合完整的prompt
        full_prompt = base_prompt + "\n" + matmul_guide
        
        return full_prompt
