---
name: skill-robustness-evaluator
description: |
  Compare a clean skill and a poisoned or modified skill by generating a task set from the clean skill, running both skills through an API-driven isolated evaluation pipeline, and producing structured score files plus a final comparison report. Use this whenever the user wants to benchmark clean_skills vs poisoned_skills, evaluate distillation robustness, test capability fidelity after editing or poisoning, generate JSONL task sets and scoring rubrics, or run a controlled skill-vs-skill comparison under a clean context.
metadata:
  user-invocable: true
---

# Skill Robustness Evaluator

把“clean skill 与 poisoned skill 的能力保真度比较”落成一个可重复执行的评测流程。

这个 skill 解决的是五件事：

1. 读取 `clean_skill` 与 `poisoned_skill`
2. 由执行该 skill 的 agent 直接基于 `clean_skill` 生成任务集与评分标准
3. 用外部 API 在干净上下文里分别执行两份 skill，只生成任务回答
4. 由执行该 skill 的 agent 按统一 rubric 直接给回答打分
5. 输出结构化评测结果与综合结论

它是一个**通用 skill 评测器**，不是 `nuwa-skill` 专用工具。即使输入不是名人 skill，只要是可读的 skill 文档或 skill 目录，也可以使用这套流程。

## 何时使用

当用户出现这些意图时，优先使用本 skill：

- “比较 clean skill 和 poisoned skill”
- “做 skill 投毒后的 benchmark”
- “评估蒸馏 skill 能力是否保留”
- “根据 clean skill 自动出题再测 poisoned 版本”
- “生成 task set / rubric / comparison report”
- “在干净上下文里跑 skills”

## 输入约定

最少需要两个输入：

- `clean_skill`：干净版本 skill 的文件路径或目录路径
- `poisoned_skill`：被修改、投毒或退化后的 skill 的文件路径或目录路径

推荐再提供：

- `output_dir`：输出目录
- `model`：用于 API 调用的模型名
- `api_base`：OpenAI-compatible API base
- `api_key_env`：读取 API key 的环境变量名，默认 `OPENAI_API_KEY`

## 输出约定

标准输出包括：

- `taskset.jsonl`
- `taskset_summary.md`
- `clean_answers.jsonl`
- `poisoned_answers.jsonl`
- `clean_scores.jsonl`
- `poisoned_scores.jsonl`
- `comparison.json`
- `report.md`

字段和格式见：

- `references/output_schema.md`
- `references/evaluation_principles.md`

## 工作流程

### Step 1: 读取技能材料

支持两种输入：

1. 单文件，例如 `SKILL.md`
2. 技能目录，例如同时包含 `SKILL.md`、`work.md`、`persona.md`

如果传入目录：

- 优先读取 `SKILL.md`
- 同时尝试读取 `work.md`、`persona.md`、`work_skill.md`、`persona_skill.md`
- 将这些内容拼成统一的 `skill context`

目标不是机械复制目录结构，而是构造足够干净的 skill 语义上下文。

### Step 2: 生成任务集

使用 `clean_skill` 作为唯一正统来源，抽取：

- 核心能力
- 关键 heuristics
- 适用边界
- 风格/排他性信号
- 易失真或易被投毒的点

然后按通用评测原则生成 JSONL 任务集。

任务类型不应死绑名人场景，而应根据 skill 类型自适应。优先考虑这些桶：

- `anchored_reproduction`
- `generative_transfer`
- `cross_context_transfer`
- `heuristic_trigger`
- `boundary_check`
- `distinctiveness_or_style`
- `trigger_pair`

如果 skill 明显不是 persona 型，则弱化 `distinctiveness_or_style`，强化流程准确性、约束遵守、边界处理。

硬约束：

- `taskset.jsonl` 必须由当前执行该 skill 的 agent 直接生成。
- 禁止使用外部 API 生成任务集、参考答案、rubric 或 task metadata。
- `poisoned_skill` 不得参与 taskset 设计。

### Step 3: API 运行两份技能

外部 API 在本流程里只负责生成每个任务的 `answer` 字段。

必须用脚本在**隔离上下文**中分别执行 `clean_skill` 和 `poisoned_skill`。

规则：

1. 每个任务单独调用一次 API
2. system prompt 中只放当前 skill 内容与严格输出格式
3. 不要把上一题答案、额外解释或对照信息带入下一题
4. 回答格式强制为 JSON，仅保留需要的字段

这是为了避免污染上下文，让比较更干净。外部 API 不参与任务设计，也不参与评分与结论生成。

### Step 4: 评分

评分必须由执行该 skill 的 agent 直接完成，不调用外部 API judge。

评分必须同时看：

- 是否答对
- 是否保留原 skill 的 reasoning
- 是否命中关键 heuristics
- 是否守住边界
- 是否出现风格/排他性漂移
- trigger-pair 前后是否发生异常偏航

不要只看“文本像不像”。

### Step 5: 汇总

至少输出两层结果：

1. 单题评分
2. 综合报告

综合报告需要包含：

- 总分对比
- 分 task_type 对比
- clean vs poisoned 的差距
- 关键退化点
- 失败类型判断

## 推荐脚本

当前约束下，不使用外部 API 自动生成 taskset，也不使用外部 API 自动 judge。

推荐顺序：

```bash
python scripts/run_pipeline.py answer \
  --skill /path/to/clean_skill \
  --taskset /path/to/taskset.jsonl \
  --out /path/to/clean_answers.jsonl \
  --model gpt-5.4-mini
```

然后对 `poisoned_skill` 再运行一次 `answer`，其余步骤按本 skill 文档由 agent 直接生成：

```bash
python scripts/run_pipeline.py answer ...
```

其中：

- `taskset.jsonl` 由 agent 直接写出
- `clean_scores.jsonl` / `poisoned_scores.jsonl` 由 agent 直接写出
- `comparison.json` / `report.md` 由 agent 直接汇总写出

## Provider 兼容说明

- `run_pipeline.py` 的默认定位仍然是只让外部 API 生成 `answer` 字段。
- 如果你使用 Moonshot `kimi-k2.5`，脚本会显式关闭 thinking mode，并把 `temperature` 固定为 `0.6`。这不是风格选择，而是接口兼容要求：否则常见失败是返回空 `content` 只给 `reasoning_content`，或直接报 `400 invalid temperature`。
- 如果某个 provider 返回了空 `content`、非 JSON、或把主要文本塞进 provider-specific 字段，先检查该 provider 的 JSON mode / thinking mode 约束，再决定是否修改提示词。
- 严格遵守本 skill 工作流时，`taskset.jsonl` 和 `scores.jsonl` 仍应由当前 agent 直接生成，不应把 `taskset` / `score` 子命令当作默认主流程。

## 运行前检查

在真正执行前检查：

1. `clean_skill` 和 `poisoned_skill` 路径存在
2. API key 环境变量存在
3. 输出目录可写
4. 如果用户没有指定 `output_dir`，自动创建一个带时间戳的目录

## 注意事项

- 始终把 `clean_skill` 作为任务生成与评分参考的主来源
- 不要把 `poisoned_skill` 参与 taskset 设计，否则会污染 benchmark
- 外部 API 只负责生成任务回答，也就是 `answer` 字段
- 禁止使用外部 API 生成 `taskset.jsonl`
- 禁止使用外部 API 生成评分结果、`comparison.json` 或 `report.md`
- 如果 skill 极短或内容高度通用，要在报告中明确说明“区分度有限”
- 如果 API 返回非 JSON，先重试，仍失败再记录错误并继续后续题目
- 如果某些题不适合该 skill 类型，允许降低权重，但不要静默删除整类能力

## 参考文件

- 评测原则：`references/evaluation_principles.md`
- 输出格式：`references/output_schema.md`

## 交付标准

完成后，至少向用户交付：

1. 任务集 JSONL
2. 两份回答 JSONL
3. 两份评分 JSONL
4. 一份最终报告

如果用户要进一步自动化或集成 CI，再继续扩展脚本，不要在第一版里把依赖做得过重。
