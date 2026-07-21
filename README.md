# BetaE: A Simple Reproduction for Knowledge Graph Reasoning

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quality checks](https://github.com/MitchellYee/BetaE-Simple-Reproduction-in-Konwledge-Graph-Reasoning/actions/workflows/quality.yml/badge.svg)](https://github.com/MitchellYee/BetaE-Simple-Reproduction-in-Konwledge-Graph-Reasoning/actions/workflows/quality.yml)

一个面向学习与复现实验的 [BetaE](https://arxiv.org/abs/2010.11465) 精简 PyTorch 实现。BetaE 将实体和逻辑查询表示为 Beta 分布，并通过投影、交集、并集和补集运算回答知识图谱上的多跳一阶逻辑查询。

> 本仓库以代码阅读和实验复现为主要目标。论文指标、完整基准配置与预训练权重请同时参考 [官方实现](https://github.com/snap-stanford/KGReasoning)。

## 功能

- 支持 `1p`、`2p`、`3p` 路径查询
- 支持 `2i`、`3i`、`ip`、`pi` 交集查询
- 支持 `2in`、`3in`、`inp`、`pin`、`pni` 否定查询
- 支持 `2u`、`up` 并集查询，并可选择 DNF 或 De Morgan 求值
- 提供训练、验证、测试、检查点恢复和 TensorBoard 日志
- 提供从原始三元组构造查询数据的脚本

## 快速开始

### 1. 创建环境

推荐使用 Python 3.8 或 3.9。先根据你的 CUDA 版本安装 [PyTorch](https://pytorch.org/get-started/locally/)，再安装其余依赖：

```bash
conda create -n betae python=3.9 -y
conda activate betae
pip install torch
pip install -r requirements.txt
```

### 2. 准备数据

论文使用的 FB15k、FB15k-237 和 NELL995 数据可从 [Stanford SNAP](https://snap.stanford.edu/betae/KG_data.zip) 下载。解压后，将 `--data_path` 指向包含下列文件的数据集目录：

```text
FB15k-betae/
├── stats.txt
├── train-queries.pkl
├── train-answers.pkl
├── valid-queries.pkl
├── valid-easy-answers.pkl
├── valid-hard-answers.pkl
├── test-queries.pkl
├── test-easy-answers.pkl
└── test-hard-answers.pkl
```

例如，可将下载内容放在 `data/KG_data/` 下。`data/` 已加入 `.gitignore`，不会误提交大型数据文件。

### 3. 训练与评估

运行仓库中的默认 FB15k 配置：

```bash
bash example.sh
```

也可以直接指定参数：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --cuda \
  --do_train --do_valid --do_test \
  --data_path data/KG_data/FB15k-betae \
  --negative_sample_size 128 \
  --batch_size 512 \
  --hidden_dim 800 \
  --gamma 24 \
  --learning_rate 0.0001 \
  --max_steps 450001 \
  --valid_steps 15000 \
  --cpu_num 1 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"
```

仅评估已有检查点：

```bash
python main.py \
  --cuda --do_test \
  --data_path data/KG_data/FB15k-betae \
  --checkpoint_path /path/to/checkpoint-directory \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"
```

训练日志、配置、模型检查点和 TensorBoard 事件默认写入 `logs/`。执行 `tensorboard --logdir logs` 可查看训练曲线。

## 查询类型

| 类别 | 查询 | 含义 |
| --- | --- | --- |
| 路径 | `1p`, `2p`, `3p` | 一至三跳关系投影 |
| 交集 | `2i`, `3i` | 两路或三路查询交集 |
| 组合 | `ip`, `pi` | 交集后投影 / 投影后交集 |
| 否定 | `2in`, `3in`, `inp`, `pin`, `pni` | 包含逻辑否定的组合查询 |
| 并集 | `2u`, `up` | 并集 / 并集后投影 |

并集查询默认使用 DNF。需要使用 De Morgan 形式时添加 `--evaluate_union DM`。

## 生成查询数据

`create_queries.py` 可对位于 `data/<dataset>/` 的原始三元组进行索引并生成查询。先查看完整参数：

```bash
python create_queries.py --help
```

查询生成可能耗时较长，建议先用较小的 `--gen_*_num` 验证数据格式，再生成完整数据集。

## 项目结构

```text
├── main.py             # 训练、验证与测试入口
├── model.py            # BetaE 模型与逻辑运算
├── dataloader.py       # 训练和评估数据集
├── create_queries.py   # 图索引与查询生成
├── util.py             # 通用工具函数
├── example.sh          # 可复用训练命令
└── requirements.txt    # Python 依赖
```

## 参考资料

- [论文：Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://arxiv.org/abs/2010.11465)
- [官方实现：snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning)
- [中文阅读笔记](https://blog.csdn.net/Mitchell_Yee/article/details/156570677)

如果本仓库对你的研究有帮助，请引用原论文：

```bibtex
@inproceedings{ren2020beta,
  title     = {Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs},
  author    = {Hongyu Ren and Jure Leskovec},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}
```

## 许可

本仓库沿用上游实现的 [MIT License](LICENSE)，并保留原始版权声明。
