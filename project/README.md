# RAG 数学推理

这是一个围绕“题型—经验知识库（Type-Experience Knowledge Base）”构建的**离线建库 + 在线检索增强推理**流水线。

在原项目基础上，这个版本加入了：

- **可学习题型抽象器**：`src/type_abstractor.py`
- **规则 / learned / hybrid 三种题型抽象策略**
- **低置信度规则回退**
- **top-k 题型候选参与检索 query 构造**
- **训练入口**：`scripts/train_type_abstractor.py`
- **学习版配置**：`configs/learned_type.yaml`

## 目录结构

```text
math_type_experience_project/
├── configs/
│   ├── default.yaml
│   └── learned_type.yaml
├── scripts/
│   ├── build_kb.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── smoke_test.py
│   └── train_type_abstractor.py
├── src/
│   ├── dataset.py
│   ├── evaluation.py
│   ├── heuristics.py
│   ├── io_utils.py
│   ├── kb_builder.py
│   ├── llm_backends.py
│   ├── pipeline.py
│   ├── retriever.py
│   ├── schema.py
│   └── type_abstractor.py
├── outputs/
├── PATCH_NOTES.md
└── requirements.txt
```

## 代码实现

### 1. 离线知识库构建

- 自动读取 `../math_datasets` 下的 `jsonl/json/csv` 文件
- 对题目做题型抽象（coarse/fine type）
- 生成结构化经验（步骤、原则、公式、易错点）
- 计算验证分数（答案有效性、题型一致性、步骤质量、经验匹配）
- 产出 `kb_entries.jsonl`、`type_taxonomy.json` 与 `retriever.pkl`

### 2. 在线检索增强推理

- 先抽象输入问题的题型
- 再做混合检索：题型相似 + 原题相似 + 经验相似 + 质量分
- 支持 `zero_shot / type_only / experience_only / full` 四种模式
- 支持检索后去冗余与多样性保留

### 3. 题型抽象升级

- `rule`：原始规则式题型抽象
- `learned`：sentence embedding + coarse/fine 层级分类器
- `hybrid`：learned 优先，低置信度时自动回退规则器
- `llm`：保留基于 LLM 的题型抽象方式

### 4. 实验评估与消融

- 生成每种模式下的逐题预测文件
- 生成 `summary.csv / summary.json`
- 保留原有 ablation 流程

## 数据集接入

把真实 `math_datasets` 文件放到上一级目录 `../math_datasets` 即可。

推荐字段：

- `question`
- `answer`
- `solution`（可选）
- `dataset`（可选）
- `split`（可选）
- `subject`（可选）
- `difficulty`（可选）

如果没有 `split` 字段，也可以直接通过文件名让程序识别：

- `*_train.jsonl`
- `*_test.jsonl`
- `*_dev.jsonl`

## 快速开始

```bash
pip install -r requirements.txt
python scripts/build_kb.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/run_ablation.py --config configs/default.yaml
```

## 使用可学习题型抽象

先训练题型抽象器：

```bash
python scripts/train_type_abstractor.py --config configs/default.yaml
```

然后使用混合策略跑实验：

```bash
python scripts/build_kb.py --config configs/learned_type.yaml
python scripts/evaluate.py --config configs/learned_type.yaml
python scripts/run_ablation.py --config configs/learned_type.yaml
```

## 切换到真实 LLM

默认配置使用 `mock` 后端，优点是可以在无 API Key 的情况下跑通全流程。

如果你要接入真实模型，把配置中的：

```yaml
llm:
  backend: mock
```

改成：

```yaml
llm:
  backend: openai_compatible
  model_name: your-model-name
  api_base: your-api-base
  api_key_env: OPENAI_API_KEY
```

并确保环境变量已设置。

## 说明

- 当前版本已经包含修改后的完整项目文件
- `PATCH_NOTES.md` 里有这次改动的说明
- `outputs/` 中保留了已有输出示例，方便你核对流程
