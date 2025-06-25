# 模型对比测试指南

本指南将帮助您对比微调前后的模型效果，评估微调的改进程度。

## 快速开始

### 1. 交互式对比测试

最简单的方式是使用交互式模式，可以实时输入问题并查看两个模型的回答对比：

```bash
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --interactive
```

**使用说明：**
- 输入问题后，系统会显示微调前后两个模型的回答
- 输入 `config` 可以调整生成参数（温度、长度等）
- 输入 `save` 可以保存当前的对比结果
- 输入 `quit` 退出程序

### 2. 批量对比测试

使用预设的问题集进行批量测试，适合系统性评估：

```bash
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --questions-file "test_questions.json" \
    --output "comparison_results.json"
```

## 详细参数说明

### 必需参数

- `--base-model`: 基础模型路径（微调前的原始模型）
- `--finetuned-model`: 微调后的模型路径（LoRA权重目录）

### 运行模式

- `--interactive`: 交互式对比模式
- `--questions-file`: 批量测试的问题文件路径
- `--output`: 结果输出文件路径

### 生成参数

- `--max-new-tokens`: 最大生成token数（默认256）
- `--temperature`: 生成温度，控制随机性（默认0.7）
- `--top-p`: nucleus采样参数（默认0.9）
- `--top-k`: top-k采样参数（默认50）
- `--repetition-penalty`: 重复惩罚系数（默认1.1）

### 设备配置

- `--device`: 指定设备（默认auto，自动选择GPU）

## 测试问题文件格式

### JSON格式

```json
{
  "questions": [
    "问题1",
    "问题2",
    "问题3"
  ]
}
```

### 文本格式

```
问题1
问题2
问题3
```

每行一个问题，空行会被忽略。

## 结果分析

### 输出格式

对比结果会保存为JSON格式，包含以下信息：

```json
[
  {
    "question": "用户问题",
    "system_prompt": null,
    "base_model": {
      "response": "基础模型的回答",
      "time": 1.23
    },
    "finetuned_model": {
      "response": "微调模型的回答",
      "time": 1.45
    }
  }
]
```

### 评估维度

建议从以下几个维度评估模型效果：

1. **回答质量**
   - 准确性：回答是否正确
   - 完整性：回答是否全面
   - 相关性：回答是否切题

2. **语言表达**
   - 流畅性：语言是否自然流畅
   - 逻辑性：回答是否有逻辑
   - 专业性：术语使用是否准确

3. **中文适应性**
   - 语法正确性
   - 文化适应性
   - 表达习惯

4. **性能指标**
   - 生成速度
   - 响应时间
   - 资源占用

## 使用示例

### 示例1：快速体验

```bash
# 交互式对比，使用默认参数
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --interactive
```

### 示例2：批量测试

```bash
# 使用预设问题集进行批量测试
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --questions-file "test_questions.json" \
    --output "results/comparison_$(date +%Y%m%d_%H%M%S).json"
```

### 示例3：调整生成参数

```bash
# 使用更保守的生成参数
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --interactive \
    --temperature 0.3 \
    --top-p 0.8 \
    --max-new-tokens 512
```

## 常见问题

### Q: 模型加载失败怎么办？

A: 检查以下几点：
1. 确认模型路径正确
2. 确认有足够的GPU内存
3. 确认依赖包已正确安装

### Q: 生成速度很慢怎么办？

A: 可以尝试：
1. 减少 `max_new_tokens` 参数
2. 使用更小的批次大小
3. 确认使用了GPU加速

### Q: 如何自定义测试问题？

A: 创建自己的问题文件：
1. JSON格式：参考 `test_questions.json`
2. 文本格式：每行一个问题
3. 可以包含特定领域的问题

### Q: 如何评估微调效果？

A: 建议关注：
1. 回答的准确性和相关性
2. 中文表达的自然程度
3. 专业术语的使用
4. 逻辑结构的清晰度

## 进阶使用

### 自定义评估指标

可以基于输出结果开发自动化评估脚本：

```python
import json

def analyze_results(results_file):
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 分析回答长度
    base_lengths = [len(r['base_model']['response']) for r in results]
    ft_lengths = [len(r['finetuned_model']['response']) for r in results]
    
    print(f"基础模型平均回答长度: {sum(base_lengths)/len(base_lengths):.1f}")
    print(f"微调模型平均回答长度: {sum(ft_lengths)/len(ft_lengths):.1f}")
    
    # 分析生成时间
    base_times = [r['base_model']['time'] for r in results]
    ft_times = [r['finetuned_model']['time'] for r in results]
    
    print(f"基础模型平均生成时间: {sum(base_times)/len(base_times):.2f}秒")
    print(f"微调模型平均生成时间: {sum(ft_times)/len(ft_times):.2f}秒")
```

### 多轮对话测试

对于需要上下文的对话场景，可以扩展测试脚本支持多轮对话评估。

## 总结

通过系统性的对比测试，您可以：

1. **量化评估**微调效果
2. **识别改进**的具体方面
3. **发现问题**并进行针对性优化
4. **验证模型**在特定场景下的表现

建议定期进行对比测试，特别是在调整训练参数或数据集后，以确保模型性能持续改进。