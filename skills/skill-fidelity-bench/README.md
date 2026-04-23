# Skill Fidelity Bench

> 衡量一个修改后或投毒后的 skill，是否还保留了 clean skill 的思维方式。

`Skill Fidelity Bench` 用来比较 `clean_skill` 与 `poisoned` 或其他修改版本 skill 的能力保真度，输出一套可复查、可复现的 benchmark 产物，而不是只给一句“看起来差不多”。

它关心的不是“回答像不像”，而是下面这些更难伪装的维度：

- 核心能力是否保住
- 推理顺序和 heuristics 是否漂移
- 边界是否还诚实
- 风格 / persona / 表达 DNA 是否被稀释
- 近义 prompt 下是否会异常偏航

这使它适合用于：

- skill distillation 后的能力保真度检查
- poisoned skill / modified skill 对 clean skill 的回归测试
- skill 发布前的 benchmark 产物生成
- clean vs modified 的可重复比较

## 为什么需要这个 skill

很多 skill 对比会在两个地方失真：

1. 只挑几道题人工比一下
2. 把 taskset、judge 和结论都外包给同一个外部模型

这个 skill 反过来做：

- `clean_skill` 是任务设计的唯一正统来源
- 外部 API 只负责生成 `answer`
- taskset、评分和结论由当前 agent 保持控制

这条边界本身，就是 benchmark 的核心价值。

## 它会产出什么

一次标准运行通常应交付：

- `taskset.jsonl`
- `taskset_summary.md`
- `clean_answers.jsonl`
- `poisoned_answers.jsonl`
- `clean_scores.jsonl`
- `poisoned_scores.jsonl`
- `comparison.json`
- `report.md`

字段定义见：

- [`references/evaluation_principles.md`](./references/evaluation_principles.md)
- [`references/output_schema.md`](./references/output_schema.md)

## 它在测什么

这套 benchmark 主要围绕五个维度组织：

- `capability_fidelity`
- `reasoning_fidelity`
- `boundary_integrity`
- `distinctiveness`
- `trigger_stability`

根据 skill 类型不同，taskset 会从这些桶里选题：

- `anchored_reproduction`
- `generative_transfer`
- `cross_context_transfer`
- `heuristic_trigger`
- `boundary_check`
- `distinctiveness_or_style`
- `trigger_pair`

如果输入 skill 不是 persona 型，风格题应弱化，流程准确性和边界处理应更强。

## 工作方式

推荐流程很简单：

1. 读取 `clean_skill` 与 `poisoned_skill`
2. 只基于 `clean_skill` 生成 taskset
3. 分别在隔离上下文里跑两份 skill，只产出 answers
4. 按 clean-skill rubric 逐题打分
5. 聚合成比较报告

这里有一条硬规则：

> 外部 API 可以生成 `answer`，但不应该负责 taskset、judge 或最终结论。

## 安装

如果你从 `TianGong-Skill` 仓库里单独使用这个 skill，最直接的安装方式是复制 skill 目录：

```bash
cp -R ./skill-fidelity-bench ~/.codex/skills/skill-fidelity-bench
```

安装后以公开名调用：

```text
skill-fidelity-bench
```

如果你的 host 支持 repo 级安装，就把这个目录作为可安装 skill 的入口即可。

## 快速开始

### 1. 准备输入

你至少需要：

- 一个 `clean_skill`
- 一个 `poisoned_skill` 或 modified skill
- 一个输出目录
- 一个 OpenAI-compatible provider 用于 answer 阶段

### 2. 由当前 agent 生成 taskset

taskset 应该只来自 clean skill，并包含：

- `question`
- `task_type`
- 精简的参考答案或回答轮廓
- 评分 rubric

### 3. 分别运行两次 answer 阶段

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/clean_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/clean_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-5.4-mini
```

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/poisoned_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/poisoned_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-5.4-mini
```

### 4. 本地评分并聚合

当前 agent 接着写出：

- `clean_scores.jsonl`
- `poisoned_scores.jsonl`
- `comparison.json`
- `report.md`

如果你已经有 score 文件，可以直接做聚合：

```bash
python scripts/run_pipeline.py report \
  --clean-skill /path/to/clean_skill \
  --poisoned-skill /path/to/poisoned_skill \
  --clean-scores /path/to/clean_scores.jsonl \
  --poisoned-scores /path/to/poisoned_scores.jsonl \
  --out-dir /path/to/output_dir
```

## 脚本边界

[`scripts/run_pipeline.py`](./scripts/run_pipeline.py) 现在提供五个子命令：

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py taskset ...
python scripts/run_pipeline.py answer ...
python scripts/run_pipeline.py score ...
python scripts/run_pipeline.py report ...
python scripts/run_pipeline.py run ...
```

推荐的理解方式是：

- `answer` 是最标准的自动化入口
- `report` 适合已有 score 文件后的汇总
- `taskset`、`score`、`run` 更偏便捷能力，不该被理解成“把整套 benchmark 全外包给 API”

如果你要严格遵守 skill 协议，taskset 和 score 仍应由当前 agent 主导。

## Provider 说明

脚本面向 OpenAI-compatible API 编写，但对 Moonshot / Kimi 做了兼容处理，因为严格 JSON 输出时它们有额外约束。

当前行为包括：

- `kimi-k2.5` 会显式关闭 thinking mode
- `kimi-k2.5` 会把 `temperature` 固定为 `0.6`
- `kimi-k2.6` 会把 `temperature` 固定为 `1.0`

如果某个 provider 返回空 `content`、把内容塞进 provider-specific reasoning 字段，或者 JSON 格式不稳定，先检查 provider 的 JSON mode 约束，再改 prompt。

## 示例产物

这个目录里已经带了一份去标识化后的实际样例：

- [`runs/jobs_vs_nobody_kimi25_20260422_211344/taskset.jsonl`](./runs/jobs_vs_nobody_kimi25_20260422_211344/taskset.jsonl)
- [`runs/jobs_vs_nobody_kimi25_20260422_211344/comparison.json`](./runs/jobs_vs_nobody_kimi25_20260422_211344/comparison.json)
- [`runs/jobs_vs_nobody_kimi25_20260422_211344/report.md`](./runs/jobs_vs_nobody_kimi25_20260422_211344/report.md)

这个样例展示的是一种常见退化模式：

- poisoned skill 没有整体崩掉
- boundary 可能仍然稳定
- 但 style、distinctiveness 和 transfer quality 已经出现可观测下降

## 目录结构

```text
skill-fidelity-bench/
├── README.md
├── SKILL.md
├── evals/
│   └── evals.json
├── references/
│   ├── evaluation_principles.md
│   └── output_schema.md
├── scripts/
│   └── run_pipeline.py
└── runs/
    └── <run-id>/
        ├── taskset.jsonl
        ├── taskset_summary.md
        ├── clean_answers.jsonl
        ├── poisoned_answers.jsonl
        ├── clean_scores.jsonl
        ├── poisoned_scores.jsonl
        ├── comparison.json
        └── report.md
```

公开 skill 名和目录名都统一为 `skill-fidelity-bench`。

## 隐私与安全

这个 skill 不应该包含硬编码凭证。

推荐做法：

- API key 一律走环境变量，比如 `OPENAI_API_KEY`
- 不要把包含私有语料的 benchmark prompt 直接提交进仓库
- 发布样例时先做去标识化

---

# English

> Benchmark whether a modified skill still preserves the clean skill's cognition.

`Skill Fidelity Bench` compares a `clean_skill` with a `poisoned` or otherwise modified version and produces a reviewable benchmark bundle instead of a vague qualitative judgment.

It is not limited to surface similarity. The benchmark focuses on:

- capability fidelity
- reasoning fidelity
- boundary integrity
- distinctiveness or style preservation when relevant
- trigger stability under near-equivalent prompts

## What It Does

A standard run is expected to generate:

- `taskset.jsonl`
- `taskset_summary.md`
- `clean_answers.jsonl`
- `poisoned_answers.jsonl`
- `clean_scores.jsonl`
- `poisoned_scores.jsonl`
- `comparison.json`
- `report.md`

Schemas and evaluation rules are documented in:

- [`references/evaluation_principles.md`](./references/evaluation_principles.md)
- [`references/output_schema.md`](./references/output_schema.md)

## Core Rule

The benchmark is built around one hard boundary:

> The external API may generate `answer` fields, but it should not design the taskset, judge the benchmark, or write the final conclusion.

The `clean_skill` is the sole source of truth for task design and scoring reference.

## Recommended Flow

1. Read the `clean_skill` and `poisoned_skill`.
2. Build the taskset from the clean skill only.
3. Run both skills in isolated answer-only API calls.
4. Score both answer sets against the clean-skill rubric.
5. Aggregate the result into `comparison.json` and `report.md`.

## Install

For local Codex-style hosts:

```bash
cp -R ./skill-fidelity-bench ~/.codex/skills/skill-fidelity-bench
```

Invoke it by its public name:

```text
skill-fidelity-bench
```

## Answer Stage Example

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/clean_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/clean_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-5.4-mini
```

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/poisoned_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/poisoned_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --model gpt-5.4-mini
```

## Script Commands

[`scripts/run_pipeline.py`](./scripts/run_pipeline.py) currently exposes:

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py taskset ...
python scripts/run_pipeline.py answer ...
python scripts/run_pipeline.py score ...
python scripts/run_pipeline.py report ...
python scripts/run_pipeline.py run ...
```

Recommended interpretation:

- `answer` is the canonical automation path
- `report` is useful when score files already exist
- `taskset`, `score`, and `run` are convenience helpers, not a license to outsource the full benchmark protocol to an API

## Provider Notes

The script targets OpenAI-compatible APIs and includes provider-specific handling for Moonshot / Kimi:

- `kimi-k2.5` disables thinking mode
- `kimi-k2.5` pins `temperature` to `0.6`
- `kimi-k2.6` pins `temperature` to `1.0`

If a provider returns empty `content`, unstable JSON, or reasoning in provider-specific fields, inspect provider JSON-mode behavior before changing prompts.

## Privacy Note

This skill should not contain hardcoded credentials.

Use environment variables for API keys and sanitize example runs before publishing them. In the current pass, no hardcoded real API key was found; the main privacy issue identified and fixed was leakage of local absolute paths in sample run artifacts.
