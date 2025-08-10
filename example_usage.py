#!/usr/bin/env python3
"""
AscendC算子Agent使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_generators.prompt_registry import PROMPT_REGISTRY
from dataset import dataset

def demonstrate_prompt_strategies():
    """演示不同的prompt策略"""
    print("=== AscendC算子Agent使用示例 ===\n")
    
    # 选择几个代表性的算子
    test_operators = [
        ("relu", "activation"),
        ("layer_norm", "normalization"), 
        ("conv_standard_2d_square_input_square_kernel", "convolution"),
        ("batched_matrix_multiplication", "matmul")
    ]
    
    for op_name, expected_category in test_operators:
        if op_name not in dataset:
            print(f"跳过不存在的算子: {op_name}")
            continue
            
        actual_category = dataset[op_name]['category']
        print(f"算子: {op_name}")
        print(f"预期类别: {expected_category}, 实际类别: {actual_category}")
        
        # 可用的prompt策略
        available_strategies = PROMPT_REGISTRY.get("ascendc", {})
        print(f"可用策略: {list(available_strategies.keys())}")
        
        # 推荐策略
        if actual_category == "activation":
            recommended = "activation_specific"
        elif actual_category == "normalization":
            recommended = "normalization_specific"
        elif actual_category == "convolution":
            recommended = "convolution_specific"
        elif actual_category == "matmul":
            recommended = "matmul_specific"
        else:
            recommended = "comprehensive"
        
        print(f"推荐策略: {recommended}")
        print("-" * 50)

def show_command_examples():
    """显示命令示例"""
    print("\n=== 命令使用示例 ===\n")
    
    examples = [
        {
            "description": "使用综合prompt生成器（推荐）",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy comprehensive --categories activation"
        },
        {
            "description": "激活函数专用策略",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy activation_specific --categories activation"
        },
        {
            "description": "归一化算子专用策略",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy normalization_specific --categories normalization"
        },
        {
            "description": "卷积算子专用策略",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy convolution_specific --categories convolution"
        },
        {
            "description": "矩阵乘法专用策略",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy matmul_specific --categories matmul"
        },
        {
            "description": "智能Agent策略",
            "command": "python generate_and_write.py --model deepseek-chat --language ascendc --strategy agent --categories all"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

def show_operator_categories():
    """显示算子类别统计"""
    print("=== 算子类别统计 ===\n")
    
    category_counts = {}
    for op_name, op_info in dataset.items():
        category = op_info['category']
        if category not in category_counts:
            category_counts[category] = []
        category_counts[category].append(op_name)
    
    for category, ops in sorted(category_counts.items()):
        print(f"{category}: {len(ops)} 个算子")
        # 显示前几个算子名称
        if len(ops) <= 5:
            print(f"  {', '.join(ops)}")
        else:
            print(f"  {', '.join(ops[:5])}...")
        print()

def show_prompt_strategy_comparison():
    """显示不同prompt策略的对比"""
    print("=== Prompt策略对比 ===\n")
    
    strategies = {
        "comprehensive": {
            "优点": "包含完整的编程约束和指导，适合所有算子类型",
            "缺点": "prompt较长，可能增加token消耗",
            "适用场景": "通用场景，特别是复杂算子"
        },
        "activation_specific": {
            "优点": "针对激活函数优化，提供专门的数值稳定性指导",
            "缺点": "仅适用于激活函数",
            "适用场景": "ReLU、Sigmoid、Tanh等激活函数"
        },
        "normalization_specific": {
            "优点": "针对归一化算子优化，包含统计信息计算指导",
            "缺点": "仅适用于归一化算子",
            "适用场景": "LayerNorm、BatchNorm等归一化算子"
        },
        "convolution_specific": {
            "优点": "针对卷积算子优化，包含tiling策略指导",
            "缺点": "仅适用于卷积算子",
            "适用场景": "各种卷积操作"
        },
        "matmul_specific": {
            "优点": "针对矩阵乘法优化，包含维度处理指导",
            "缺点": "仅适用于矩阵乘法",
            "适用场景": "矩阵乘法和批量矩阵乘法"
        },
        "agent": {
            "优点": "自动选择合适的策略，使用简单",
            "缺点": "可能不是最优选择",
            "适用场景": "快速原型开发"
        }
    }
    
    for strategy, info in strategies.items():
        print(f"策略: {strategy}")
        print(f"  优点: {info['优点']}")
        print(f"  缺点: {info['缺点']}")
        print(f"  适用场景: {info['适用场景']}")
        print()

def main():
    """主函数"""
    print("AscendC算子Agent使用指南")
    print("=" * 60)
    
    # 显示算子类别统计
    show_operator_categories()
    
    # 显示prompt策略对比
    show_prompt_strategy_comparison()
    
    # 演示prompt策略
    demonstrate_prompt_strategies()
    
    # 显示命令示例
    show_command_examples()
    
    print("=" * 60)
    print("更多信息请参考 ASCENDC_AGENT_README.md")

if __name__ == "__main__":
    main()
