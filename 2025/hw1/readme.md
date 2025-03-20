# Minimum Implementation of O1
# 作业目的
了解大模型强化学习算法的基本原理
掌握大模型强化学习训练的代码框架及其使用（包括多卡训练、超参数的设置、训练过程的监控等）
体会数据在大模型训练中的重要性，分析不同的数据选择策略对训练效果的影响
通过实验分析，比较监督学习（SFT）及强化学习各自的优劣
对inference-time scaling建立更深的理解Assignment Details

# 作业内容
- 基于提供的RL代码，在数学推理领域，以Qwen2.5-Math-1.5B为起点，通过强化学习进行模型能力的提升
- 提供多种类型的数据集
  - 简单难度（MATH Level1-2）：shortcot + longcot
  - 中等难度（MATH Level3-5）：shortcot + longcot
  - 困难难度（AIME）：shortcot + longcot
- 整个过程中，你需要：
  - 学习在多GPU环境下进行强化学习的训练，运行GRPO算法
  - 自行设计奖励机制，实现Loss函数
  - （可选）改进强化学习算法本身
  - 探究不同数据的选择对训练效果的影响
- 整个过程中你需要注意训练的效率，用更少的GPU卡时得到更好效果的模型
- 评估时，会提供4个数学推理benchmark

# 问题探究
在整个作业中，需要思考如下的RQ（科研问题），并设计相关的对比实验详细分析：
- SFT与RL相比，各自的优势和劣势是什么，可从以下方面探讨：
  - 算法效率
  - 泛化性
  - ...
- 在SFT&RL阶段，数据对训练的影响分别有哪些？
- 如何进行高质量的数据筛选？
# 作业要点
## 环境安装
```
conda create -n <myenv> python==3.10.12
cd sjtu_cs2916_25_homework1
pip install -r requirements.txt
```
## 文件组成
```
data
├── eval # 包含各种用于评估的数据集
├── output # 可以用于存放evaluation和rl过程中的输出样本
├── train # 用于存放各种训练样本,数据难度已经标明
├── prepare.py # 用于将jsonl格式数据转化为RL程序能读取的数据
scripts
├── sft_cot.sh # 一个启动短cot sft的示例
├── grpo.sh # 启动rl的示例
src    
├── cli    # 训练的启动入口文件
├── evaluation    # 评估sft结果的文件
├── ...
```
## 代码实现
你需要实现`src/cli/server_rm.py`中的`get_score`函数和`src/models/loss.py`中的`PolicyLoss`模块（此部分只需要实现主要的loss，不需要GRPO loss公式中的KL Loss）。
## 训练&评估
对于sft，可以直接`bash scipts/sft_cot.sh`即可。对于RL，需要先实现上述代码，然后运行`scripts/grpo.sh`.
RL会在训练中在线的评估模型的性能，sft则需要使用`bash scripts/eval.sh`对保存的模型文件进行评估。
## 结果可视化
使用Wandb对实验结果进行可视化，动态跟踪实验进展。
参考：https://wandb.ai/xuefengli0301/sjtu_cs2916_baseline?nw=nwuserxuefengli0301
## Baseline

| Model          | GSM8k | MATH | AMC23 | OlympiadBench |
|----------------|-------|------|-------|---------------|
| Short CoT SFT  | 71.6  | 44   | 30    | 9.6           |
| Long CoT SFT   | 66.5  | 40.4 | 20    | 11.5          |
| RL(GRPO)       | 74.1  | 61.3 | 50    | 17.4          |

## 计算资源估计
均在4*H800上测试。

Short CoT SFT：
- 训练：～5min
- 评估：～5min

Long CoT SFT:
- 训练: ～30min
- 评估：~20min

RL: 大约7h，300step收敛

## Notes
1. 对于SFT，可供调整的参数包括Batchsize, Epoch, Learning Rate。同时对于Long CoT的训练和评估，需要注意max_len参数的选择。
2. 对于GRPO，建议调整的超参数包括
- Rollout Batch size
- n samples per prompt
- Temperature
- 同时注意train batch size需要能整除 rollout batch size * n samples per prompt 其他参数不建议调整。
3. RL(特别是GRPO)对于问题的难度比较敏感，可以使用`./data/prepare.py`中构建难度更加合理的数据集。
4. 尽可能多关注RL的各种曲线，RL不需要像SFT一样跑到最后，如果模型性能出现明显下降或一直处于平台期即可关闭。


# 作业提交（Canvas 提交）
提交格式，一个zip文件包含以下内容：
```
学号_名字/
├── README.md # readme需要介绍简单介绍你修改/实现的代码部分
                # readme包含各个实验和该对应的wandb链接
├── report.pdf # 实验报告
scripts
├── sft_cot.sh # 一个启动短cot sft的示例
├── grpo.sh # 启动rl的示例
src    
├── cli    # 训练的启动入口文件
├── evaluation    # 评估sft结果的文件
├── ...
```

# 评分
- 100: 实现所有缺失的代码，如reward函数，强化学习的loss，使得SFT和RL能够稳定训练；通过实验，总结出数据筛选方法，并在SFT和RL的结果中均超过Baseline。
- 95：实现所有缺失的代码，如reward函数，强化学习的loss，使得SFT和RL能够稳定训练，并且实验结果达到Baseline。
- 90：实现所有缺失的代码，并且能够实现SFT和RL的稳定训练，如训练reward上涨，准确率不下降等。
- 85及以下：不能完成全部缺失代码的实现。

如果您的结果可以通过提交的文件进行确认，但是通过Canvas提交的代码存在问题，例如格式不正确、无法在适当的时间内执行等，您的成绩将被扣除5分（例如：100分变为95分，或95分变为90分）。


# 参考论文：
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- O1 Replication Journey: A Strategic Progress Report -- Part 1
- O1 Replication Journey--Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?
- O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning
- LIMO: Less is More for Reasoning
- LIMR: Less is More for RL Scaling
