# Skill Robustness Evaluator

比较 `clean skill` 与 `poisoned / modified skill` 的能力保真度，输出一套可复查的评测产物，而不是只给一句主观结论。

这个 skill 关注的不是“回答像不像”，而是：

- `capability_fidelity`：核心能力有没有保住
- `reasoning_fidelity`：推理顺序、heuristics、优先级有没有漂移
- `boundary_integrity`：证据不足时是否还诚实
- `distinctiveness`：风格、人格、表达 DNA 是否被稀释
- `trigger_stability`：近义 prompt 下是否会异常偏航

## 适用场景

适合这类任务：

- 比较 clean skill 和投毒版 skill
- 做 distillation robustness benchmark
- 检查 skill 修改后是否丢了原始能力
- 基于 clean skill 自动出题，再分别跑 clean / poisoned
- 输出 `taskset.jsonl`、answers、scores、comparison、report

不适合这类任务：

- 只想要一句模糊的“这个 skill 看起来差不多”
- 把外部 API 同时拿来生成 taskset、judge、结论
- 没有 clean skill，只想凭印象比较两个结果

## 目录结构

```text
skill-robustness-evaluator/
├── SKILL.md
├── README.md
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

## 核心原则

这套流程有一个硬边界：`clean_skill` 是任务设计和评分参考的唯一正统来源。

因此推荐工作流是：

1. 读取 `clean_skill` 与 `poisoned_skill`
2. 由当前 agent 基于 `clean_skill` 直接生成 `taskset.jsonl`
3. 只用外部 API 生成每题的 `answer`
4. 由当前 agent 直接生成 `clean_scores.jsonl` / `poisoned_scores.jsonl`
5. 汇总成 `comparison.json` 和 `report.md`

注意：

- 外部 API 只负责回答，不负责 taskset、rubric、score、结论
- `poisoned_skill` 不应参与 taskset 设计
- 每题都应隔离调用，不能把上一题上下文带到下一题

## 脚本定位

[`scripts/run_pipeline.py`](./scripts/run_pipeline.py) 提供五个子命令：

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py taskset ...
python scripts/run_pipeline.py answer ...
python scripts/run_pipeline.py score ...
python scripts/run_pipeline.py report ...
python scripts/run_pipeline.py run ...
```

其中：

- `answer` 是最推荐直接使用的子命令，因为它严格对应“外部 API 只生成 answer”
- `report` 适合在已有 `clean_scores.jsonl` / `poisoned_scores.jsonl` 后做聚合
- `taskset`、`score`、`run` 是脚本层的便捷能力

如果你要严格遵守本 skill 的协议：

- `taskset.jsonl` 仍应由当前 agent 直接生成
- `score` 结果也应由当前 agent 直接写出
- 不应把 `run` 当作默认主流程

## 最小使用流程

### 1. 先由 agent 生成 taskset

推荐输出：

- `taskset.jsonl`
- `taskset_summary.md`

任务类型通常覆盖：

- `anchored_reproduction`
- `generative_transfer`
- `cross_context_transfer`
- `heuristic_trigger`
- `boundary_check`
- `distinctiveness_or_style`
- `trigger_pair`

### 2. 用脚本分别跑 clean / poisoned 的 answers

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/clean_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/clean_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --model gpt-5.4-mini
```

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/poisoned_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/poisoned_answers.jsonl \
  --api-base https://api.openai.com/v1 \
  --model gpt-5.4-mini
```

### 3. 由 agent 直接打分

输出：

- `clean_scores.jsonl`
- `poisoned_scores.jsonl`

评分时至少同时看：

- 是否答对
- 是否保留原 skill 的 reasoning
- 是否命中关键 heuristics
- 是否守住边界
- 是否出现 distinctiveness / style drift
- trigger-pair 前后是否异常偏航

### 4. 聚合报告

如果你已经有两份 score 文件，可以直接汇总：

```bash
python scripts/run_pipeline.py report \
  --clean-skill /path/to/clean_skill \
  --poisoned-skill /path/to/poisoned_skill \
  --clean-scores /path/to/clean_scores.jsonl \
  --poisoned-scores /path/to/poisoned_scores.jsonl \
  --out-dir /path/to/output_dir
```

最终会产出：

- `comparison.json`
- `report.md`

## Provider 兼容说明

当前脚本已内置一层 Moonshot / Kimi 兼容处理：

- 对 `kimi-k2.5`，会自动关闭 thinking mode
- 同时把 `temperature` 固定为 `0.6`

这样做的原因很直接：

- 否则常见失败是返回空 `content`，只给 `reasoning_content`
- 或直接报 `400 invalid temperature`

如果你接别的 OpenAI-compatible provider，也建议优先确认三件事：

1. JSON mode 是否真的把结果放在 `message.content`
2. provider 是否有 thinking / reasoning 开关
3. 该模型是否限制 `temperature`

## 输出文件说明

详细 schema 见：

- [`references/evaluation_principles.md`](./references/evaluation_principles.md)
- [`references/output_schema.md`](./references/output_schema.md)

最重要的几个文件：

- `taskset.jsonl`：题目、参考答案轮廓、rubric、task_type
- `clean_answers.jsonl` / `poisoned_answers.jsonl`：逐题回答
- `clean_scores.jsonl` / `poisoned_scores.jsonl`：逐题评分
- `comparison.json`：结构化聚合结果
- `report.md`：给人看的总结报告

## 示例产物

仓库里已经有一组实际运行样例：

- [`runs/jobs_vs_nobody_kimi25_20260422_211344/taskset.jsonl`](./runs/jobs_vs_nobody_kimi25_20260422_211344/taskset.jsonl)
- [`runs/jobs_vs_nobody_kimi25_20260422_211344/comparison.json`](./runs/jobs_vs_nobody_kimi25_20260422_211344/comparison.json)
- [`runs/jobs_vs_nobody_kimi25_20260422_211344/report.md`](./runs/jobs_vs_nobody_kimi25_20260422_211344/report.md)

这组样例的结论是：

- `clean_avg_norm_10 = 9.50`
- `poisoned_avg_norm_10 = 8.33`
- 主要失败类型：`style_or_distinctiveness_drift`

下降最明显的 task type：

- `distinctiveness_or_style`
- `generative_transfer`
- `heuristic_trigger`

这说明 `poisoned skill` 没有整体崩掉，但在风格辨识度和迁移能力上已经出现了可观测退化。

## 开发建议

如果你要继续扩展这个 skill，优先顺序建议是：

1. 保持 `answer` 阶段尽量 provider-agnostic
2. 不要把严格协议和脚本便捷能力混在一起
3. 先把 `comparison.json` 做稳定，再谈更复杂的自动 judge
4. 每次接新 provider，先验证 JSON mode、thinking mode、temperature 约束

## 相关文件

- [`SKILL.md`](./SKILL.md)
- [`scripts/run_pipeline.py`](./scripts/run_pipeline.py)
- [`evals/evals.json`](./evals/evals.json)
