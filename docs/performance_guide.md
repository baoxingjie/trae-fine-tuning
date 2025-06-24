# 性能优化与调试指南

## 📋 目录

1. [性能监控](#性能监控)
2. [内存优化](#内存优化)
3. [训练速度优化](#训练速度优化)
4. [GPU利用率优化](#gpu利用率优化)
5. [数据加载优化](#数据加载优化)
6. [模型优化技术](#模型优化技术)
7. [调试工具和技巧](#调试工具和技巧)
8. [性能基准测试](#性能基准测试)
9. [故障诊断](#故障诊断)
10. [最佳实践总结](#最佳实践总结)

## 📊 性能监控

### 系统监控工具

#### 1. GPU监控

```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# 详细GPU信息
nvidia-smi -l 1

# GPU利用率历史
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1

# 保存监控日志
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > gpu_monitor.csv
```

#### 2. 系统资源监控

```bash
# CPU和内存监控
htop

# 磁盘I/O监控
iotop

# 网络监控
iftop

# 综合系统监控
top -p $(pgrep -d',' python)
```

#### 3. Python性能监控

```python
# 内存使用监控
import psutil
import GPUtil

def monitor_resources():
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 内存使用
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used = memory.used / (1024**3)  # GB
    
    # GPU使用率
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_util = gpu.load * 100
        gpu_memory = gpu.memoryUtil * 100
        
    print(f"CPU: {cpu_percent}%, Memory: {memory_percent}% ({memory_used:.1f}GB)")
    print(f"GPU: {gpu_util}%, GPU Memory: {gpu_memory}%")
```

### 训练监控

#### TensorBoard集成

```python
# 在训练脚本中添加详细监控
from torch.utils.tensorboard import SummaryWriter
import time

class TrainingMonitor:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.start_time = time.time()
        
    def log_metrics(self, step, metrics):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            
    def log_system_metrics(self, step):
        # GPU监控
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
            self.writer.add_scalar('System/GPU_Memory_GB', gpu_memory, step)
            self.writer.add_scalar('System/GPU_Cached_GB', gpu_cached, step)
            
        # 训练速度
        elapsed_time = time.time() - self.start_time
        samples_per_second = step / elapsed_time if elapsed_time > 0 else 0
        self.writer.add_scalar('System/Samples_Per_Second', samples_per_second, step)
```

## 💾 内存优化

### 1. 梯度检查点 (Gradient Checkpointing)

```yaml
# training_config.yaml
optimization:
  gradient_checkpointing: true
  
# 或在代码中启用
model.gradient_checkpointing_enable()
```

**效果**: 减少50-80%的GPU内存使用，但会增加20-30%的训练时间。

### 2. 混合精度训练

```yaml
# 使用bfloat16（推荐）
training:
  bf16: true
  
# 或使用float16
training:
  fp16: true
  fp16_opt_level: "O1"  # 可选: O0, O1, O2, O3
```

**内存节省**: 约50%的激活内存和模型参数内存。

### 3. LoRA (Low-Rank Adaptation)

```yaml
lora:
  use_lora: true
  r: 8                    # rank，越小内存越少
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: 
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**内存节省**: 减少90%以上的可训练参数。

### 4. 量化技术

```python
# 4-bit量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 5. DeepSpeed ZeRO

```json
// deepspeed_config.json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 1000
    }
  }
}
```

### 内存使用估算

```python
def estimate_memory_usage(model_size_b, sequence_length, batch_size, precision="fp16"):
    """
    估算训练内存使用量
    
    Args:
        model_size_b: 模型参数数量（十亿）
        sequence_length: 序列长度
        batch_size: 批次大小
        precision: 精度类型
    """
    # 参数内存 (GB)
    if precision == "fp32":
        param_memory = model_size_b * 4
    elif precision in ["fp16", "bf16"]:
        param_memory = model_size_b * 2
    elif precision == "int8":
        param_memory = model_size_b * 1
    elif precision == "int4":
        param_memory = model_size_b * 0.5
    
    # 梯度内存 (与参数相同)
    gradient_memory = param_memory
    
    # 优化器状态 (Adam需要2倍参数内存)
    optimizer_memory = param_memory * 2
    
    # 激活内存 (近似)
    activation_memory = batch_size * sequence_length * model_size_b * 0.001
    
    total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    print(f"参数内存: {param_memory:.1f} GB")
    print(f"梯度内存: {gradient_memory:.1f} GB")
    print(f"优化器内存: {optimizer_memory:.1f} GB")
    print(f"激活内存: {activation_memory:.1f} GB")
    print(f"总内存需求: {total_memory:.1f} GB")
    
    return total_memory

# 示例：Qwen-7B模型
estimate_memory_usage(7, 2048, 4, "bf16")
```

## ⚡ 训练速度优化

### 1. 批次大小优化

```python
# 自动寻找最优批次大小
def find_optimal_batch_size(model, tokenizer, max_batch_size=32):
    for batch_size in range(1, max_batch_size + 1):
        try:
            # 创建测试数据
            inputs = tokenizer(["测试文本"] * batch_size, 
                             return_tensors="pt", 
                             padding=True, 
                             truncation=True)
            
            # 前向传播测试
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"批次大小 {batch_size}: 成功")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"批次大小 {batch_size}: 内存不足")
                return batch_size - 1
            else:
                raise e
    
    return max_batch_size
```

### 2. 梯度累积

```yaml
# 等效大批次训练
training:
  per_device_train_batch_size: 2      # 实际批次
  gradient_accumulation_steps: 8       # 累积步数
  # 等效批次大小 = 2 * 8 * GPU数量
```

### 3. 数据并行优化

```bash
# 多GPU训练
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         scripts/train.py \
         --config config/training_config.yaml

# 多节点训练
torchrun --nnodes=2 \
         --nproc_per_node=4 \
         --node_rank=0 \
         --master_addr="192.168.1.100" \
         --master_port=29500 \
         scripts/train.py \
         --config config/training_config.yaml
```

### 4. 编译优化

```python
# PyTorch 2.0 编译优化
model = torch.compile(model, mode="reduce-overhead")

# 或者
model = torch.compile(model, mode="max-autotune")
```

### 5. 数据加载优化

```yaml
data:
  dataloader_num_workers: 8           # 数据加载进程数
  dataloader_pin_memory: true         # 固定内存
  dataloader_prefetch_factor: 2       # 预取因子
  preprocessing_num_workers: 16       # 预处理进程数
```

## 🎯 GPU利用率优化

### 1. GPU利用率监控

```python
import subprocess
import time

def monitor_gpu_utilization(duration=60):
    """监控GPU利用率"""
    start_time = time.time()
    utilizations = []
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            util = int(result.stdout.strip())
            utilizations.append(util)
            print(f"GPU利用率: {util}%")
        
        time.sleep(1)
    
    avg_util = sum(utilizations) / len(utilizations)
    print(f"平均GPU利用率: {avg_util:.1f}%")
    return avg_util
```

### 2. 提高GPU利用率的方法

#### 增加批次大小
```yaml
training:
  per_device_train_batch_size: 8      # 尽可能大
  gradient_accumulation_steps: 4       # 保持等效批次大小
```

#### 优化序列长度
```yaml
data:
  max_seq_length: 2048                # 根据GPU内存调整
  pack_sequences: true                # 打包短序列
```

#### 异步数据加载
```python
class AsyncDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        first = True
        for next_batch in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_batch = {k: v.cuda(non_blocking=True) 
                            for k, v in next_batch.items()}
            
            if not first:
                yield batch
            else:
                first = False
                
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = next_batch
            
        yield batch
```

## 📁 数据加载优化

### 1. 数据预处理优化

```python
class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预先加载和缓存数据
        self.data = self._load_and_cache_data(data_path)
        
    def _load_and_cache_data(self, data_path):
        cache_path = data_path + ".cache"
        
        if os.path.exists(cache_path):
            print("加载缓存数据...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print("处理原始数据...")
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # 预处理数据
                processed_item = self._preprocess_item(item)
                data.append(processed_item)
        
        # 保存缓存
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        return data
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### 2. 内存映射文件

```python
import mmap

class MemoryMappedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.line_offsets = []
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode('utf-8'))
    
    def __len__(self):
        return len(self.line_offsets)
    
    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            return json.loads(line)
```

### 3. 多进程数据加载

```python
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method

# 设置多进程启动方法
set_start_method('spawn', force=True)

# 优化的DataLoader配置
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,              # 根据CPU核心数调整
    pin_memory=True,            # 固定内存，加速GPU传输
    prefetch_factor=2,          # 预取因子
    persistent_workers=True,    # 保持worker进程
    drop_last=True             # 丢弃最后不完整的批次
)
```

## 🔧 模型优化技术

### 1. 模型剪枝

```python
import torch.nn.utils.prune as prune

def prune_model(model, pruning_ratio=0.2):
    """结构化剪枝"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    return model

# 应用剪枝
pruned_model = prune_model(model, 0.1)  # 剪枝10%的参数
```

### 2. 知识蒸馏

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### 3. 动态批次大小

```python
class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens=4096):
        self.dataset = dataset
        self.max_tokens = max_tokens
        
    def __iter__(self):
        batch = []
        current_tokens = 0
        
        for idx in range(len(self.dataset)):
            item_length = len(self.dataset[idx]['input_ids'])
            
            if current_tokens + item_length > self.max_tokens and batch:
                yield batch
                batch = []
                current_tokens = 0
            
            batch.append(idx)
            current_tokens += item_length
        
        if batch:
            yield batch
```

## 🛠️ 调试工具和技巧

### 1. 性能分析工具

```python
# PyTorch Profiler
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("model_training"):
        # 训练代码
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 保存分析结果
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2. 内存泄漏检测

```python
import gc
import torch

def detect_memory_leak():
    """检测内存泄漏"""
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    
    # 检查GPU内存
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"GPU内存 - 已分配: {allocated/1024**3:.2f}GB, 已保留: {reserved/1024**3:.2f}GB")
    
    # 检查Python对象
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"进程内存: {memory_info.rss/1024**3:.2f}GB")

# 在训练循环中定期调用
for step, batch in enumerate(dataloader):
    # 训练代码
    
    if step % 100 == 0:
        detect_memory_leak()
```

### 3. 梯度监控

```python
def monitor_gradients(model):
    """监控梯度"""
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # 检查梯度异常
            if torch.isnan(param.grad).any():
                print(f"警告: {name} 包含NaN梯度")
            if torch.isinf(param.grad).any():
                print(f"警告: {name} 包含无穷梯度")
    
    total_norm = total_norm ** (1. / 2)
    print(f"梯度范数: {total_norm:.4f}, 参数数量: {param_count}")
    
    return total_norm
```

## 📈 性能基准测试

### 1. 训练速度基准

```python
import time
from collections import defaultdict

class TrainingBenchmark:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def benchmark_training_step(self, model, batch, optimizer):
        """基准测试单个训练步骤"""
        torch.cuda.synchronize()
        start_time = time.time()
        
        # 前向传播
        forward_start = time.time()
        outputs = model(**batch)
        loss = outputs.loss
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # 反向传播
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # 优化器步骤
        optimizer_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        total_time = time.time() - start_time
        
        # 记录指标
        self.metrics['forward_time'].append(forward_time)
        self.metrics['backward_time'].append(backward_time)
        self.metrics['optimizer_time'].append(optimizer_time)
        self.metrics['total_time'].append(total_time)
        
        return {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optimizer_time': optimizer_time,
            'total_time': total_time
        }
    
    def report(self):
        """生成性能报告"""
        for metric, values in self.metrics.items():
            avg_time = sum(values) / len(values)
            print(f"{metric}: {avg_time:.4f}s (平均)")
```

### 2. 吞吐量测试

```python
def measure_throughput(model, dataloader, num_steps=100):
    """测量训练吞吐量"""
    model.train()
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_samples = 0
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
            
        batch_size = batch['input_ids'].size(0)
        total_samples += batch_size
        
        # 模拟训练步骤
        with torch.no_grad():
            outputs = model(**batch)
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    throughput = total_samples / elapsed_time
    print(f"吞吐量: {throughput:.2f} samples/second")
    print(f"总样本: {total_samples}, 总时间: {elapsed_time:.2f}s")
    
    return throughput
```

## 🔍 故障诊断

### 常见问题诊断清单

#### 1. 内存问题

```bash
# 检查GPU内存
nvidia-smi

# 检查系统内存
free -h

# 检查进程内存使用
ps aux | grep python
```

**解决方案**:
- 减少批次大小
- 启用梯度检查点
- 使用LoRA或量化
- 增加梯度累积步数

#### 2. 训练速度慢

```python
# 检查GPU利用率
def check_gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

util = check_gpu_utilization()
if util < 80:
    print("GPU利用率低，可能的原因:")
    print("- 批次大小太小")
    print("- 数据加载瓶颈")
    print("- CPU预处理瓶颈")
```

#### 3. 梯度问题

```python
def diagnose_gradients(model):
    """诊断梯度问题"""
    has_nan = False
    has_inf = False
    zero_grad_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"NaN梯度: {name}")
            
            if torch.isinf(param.grad).any():
                has_inf = True
                print(f"无穷梯度: {name}")
            
            if param.grad.abs().max() < 1e-8:
                zero_grad_params += 1
    
    print(f"梯度诊断结果:")
    print(f"- 包含NaN: {has_nan}")
    print(f"- 包含无穷: {has_inf}")
    print(f"- 零梯度参数: {zero_grad_params}/{total_params}")
```

## 🏆 最佳实践总结

### 1. 内存优化优先级

1. **启用LoRA** - 最大的内存节省
2. **使用混合精度** - 50%内存节省
3. **梯度检查点** - 大幅减少激活内存
4. **优化批次大小** - 平衡内存和速度
5. **量化技术** - 进一步压缩模型

### 2. 速度优化优先级

1. **增加批次大小** - 提高GPU利用率
2. **多GPU训练** - 线性加速
3. **优化数据加载** - 消除I/O瓶颈
4. **编译优化** - PyTorch 2.0编译
5. **异步处理** - 重叠计算和通信

### 3. 监控指标

- **GPU利用率**: 目标 >85%
- **GPU内存使用**: 目标 >90%
- **训练速度**: samples/second
- **梯度范数**: 检测梯度爆炸/消失
- **损失曲线**: 监控训练进度

### 4. 调试流程

1. **基线测试** - 记录初始性能
2. **逐步优化** - 一次改变一个参数
3. **性能监控** - 持续监控关键指标
4. **问题定位** - 使用profiling工具
5. **验证改进** - 确认优化效果

---

**记住**: 性能优化是一个迭代过程，需要根据具体硬件和任务特点进行调整。始终在优化前后进行基准测试，确保改进的有效性。