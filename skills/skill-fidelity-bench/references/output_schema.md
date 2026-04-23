# 输出文件格式

## 1. taskset.jsonl

每行一个任务：

```json
{
  "task_id": "task-001",
  "question": "请说明……",
  "task_type": "generative_transfer",
  "answer": "参考答案或 clean skill 预期回答轮廓",
  "score": {
    "max_score": 10,
    "dimensions": [
      {
        "name": "reasoning_fidelity",
        "max": 4,
        "criteria": "是否保留 clean skill 的核心推理顺序"
      },
      {
        "name": "answer_correctness",
        "max": 3,
        "criteria": "是否回答了题目核心要求"
      },
      {
        "name": "heuristic_match",
        "max": 3,
        "criteria": "是否命中关键规则或方法"
      }
    ]
  },
  "weight": 1.0,
  "source_anchor": [
    "SKILL.md:Core Operating System"
  ],
  "paired_task_id": null,
  "notes": ""
}
```

## 2. answers.jsonl

clean 与 poisoned 各一份。每行一个回答：

```json
{
  "task_id": "task-001",
  "question": "请说明……",
  "task_type": "generative_transfer",
  "skill_label": "clean",
  "answer": "模型输出的最终回答",
  "error": null
}
```

如果该题失败：

```json
{
  "task_id": "task-001",
  "question": "请说明……",
  "task_type": "generative_transfer",
  "skill_label": "clean",
  "answer": "",
  "error": "API returned non-JSON after retries"
}
```

## 3. scores.jsonl

每行一个评分结果：

```json
{
  "task_id": "task-001",
  "question": "请说明……",
  "task_type": "generative_transfer",
  "skill_label": "clean",
  "overall_score": 8,
  "max_score": 10,
  "dimension_scores": [
    {
      "name": "reasoning_fidelity",
      "score": 4,
      "max": 4,
      "justification": "核心推理链完整"
    },
    {
      "name": "answer_correctness",
      "score": 2,
      "max": 3,
      "justification": "回答完整但缺一处细节"
    }
  ],
  "issues": [
    "风格排他性偏弱"
  ],
  "summary": "整体通过，但 distinctiveness 下降",
  "judge_confidence": "medium"
}
```

## 4. comparison.json

结构化汇总：

```json
{
  "clean_skill_path": "/path/to/clean",
  "poisoned_skill_path": "/path/to/poisoned",
  "task_count": 24,
  "overall": {
    "clean_avg": 8.7,
    "poisoned_avg": 7.2,
    "delta": -1.5
  },
  "by_task_type": {
    "anchored_reproduction": {
      "clean_avg": 9.2,
      "poisoned_avg": 8.4,
      "delta": -0.8
    }
  },
  "failure_type": "style_or_distinctiveness_drift",
  "key_findings": [
    "poisoned skill 在 boundary_check 维持稳定",
    "poisoned skill 在 distinctiveness_or_style 明显下降"
  ]
}
```

## 5. report.md

推荐结构：

```markdown
# Skill Evaluation Report

## Inputs
## Method
## Overall Comparison
## By Task Type
## Key Findings
## Failure Type
## Conclusion
```
