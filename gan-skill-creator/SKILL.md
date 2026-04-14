---
name: gan-skill-creator
displayName: GAN Skill Creator
description: |
  Create high-purity AI agent skills using GAN-based knowledge distillation.
  
  Use this skill when:
  - You want to capture someone's thinking patterns into a reusable Skill
  - You need to distill expert knowledge with high purity (>90%)
  - You want to test and iterate on Skills with quantitative benchmarks
  
  The system will guide you through: intent capture → data collection → GAN distillation → skill drafting → testing → iteration
version: 1.0.0
---

# GAN Skill Creator

Create reusable Claude Skills from people's expertise using adversarial training to ensure high-purity knowledge distillation.

## Overview

This is a complete workflow for turning someone (a person, an expert, a team) into a callable Skill that Claude can use. The key innovation: **GAN-based distillation** ensures the Skill captures genuine expertise patterns, not just surface-level behavior.

The process:

1. **Intent Capture** - What should this Skill do?
2. **Data Collection** - Gather internet sources about the person
3. **Knowledge Extraction** - Claude extracts structured knowledge
4. **GAN Distillation** - Adversarial training creates high-purity representation
5. **Skill Drafting** - Generate SKILL.md from distilled knowledge
6. **Testing** - Run realistic test cases
7. **Evaluation** - Quantitative + qualitative assessment
8. **Iteration** - Improve based on feedback
9. **Description Optimization** - Tune when Skill triggers

## Your Job

Figure out where the user is in this process and help them progress:

- If they say "I want to create a Skill for X", help them narrow down intent, collect data, and walk through the full pipeline.
- If they say "I have data, let me run the GAN distillation", jump straight to step 4.
- If they say "I have a draft Skill, make it better", go to step 6 onwards.

Be flexible. If they want to skip testing and just vibe, that's fine too.

## The GAN Innovation

Unlike regular knowledge distillation (which just compresses), GAN-based distillation uses adversarial training:

- **Generator**: Learns to compress expertise patterns while preserving key insights
- **Discriminator**: Learns to evaluate whether distilled knowledge is "pure" (matches original intent)
- **Result**: High-purity representations that actually capture how someone thinks

Purity scores tell you: "Did we preserve the real expertise, or just surface mimicry?"

## Next Steps

1. Tell me who you want to distill into a Skill
2. I'll help you collect data
3. We'll run GAN distillation
4. Generate your Skill
5. Test and iterate

Let's build something great!