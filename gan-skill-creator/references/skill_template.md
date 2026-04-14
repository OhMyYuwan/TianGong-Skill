# Skill Template

Use this template to generate SKILL.md

## Variables to Replace

- `skill-name`: lowercase-with-dashes
- `Display Name`: Friendly name
- `[description]`: When and why to use
- `[core principles]`: Key ideas
- `[framework]`: Decision-making approach

## Template

```markdown
---
name: skill-name
displayName: Display Name
description: |
  When to use: Brief description of when Claude should use this skill.
  
  What it does: Clear explanation of capabilities.
  
  Make it "pushy" - include specific trigger contexts:
  - Mention domains where this applies
  - Give examples of relevant user phrases
  - Encourage usage when helpful
version: 1.0.0
---

# [Display Name]

## Overview
Clear, concise explanation of what this skill enables.

## Core Principles
- Principle 1: Why it matters
- Principle 2: Why it matters  
- Principle 3: Why it matters

## Decision Framework

Step-by-step approach:

1. **Identify**: Analyze the problem
2. **Consider**: Think about constraints
3. **Decide**: Make decision based on [framework]
4. **Execute**: Implement with focus on [values]

## Example Workflows

### Example 1: [Scenario]
**User asks**: [Realistic prompt]

**Skill does**:
1. Break down the problem
2. Consider alternatives
3. Recommend approach
4. Explain reasoning

**Output**: [What user gets]

## Communication Style
- Direct and clear
- Emphasize [core principle]
- Use analogies from [domain]
- Challenge assumptions respectfully

## Edge Cases
- Edge case 1? → Response
- Edge case 2? → Response

## Success Criteria
The skill works when:
- Output shows principled reasoning
- Recommendations are grounded in fundamentals
- User finds it genuinely useful

Tips
Keep under 500 lines
Use imperative form
Explain WHY, not just WHAT
Be specific enough to be useful, general enough to apply broadly
Test examples with real prompts
Anti-patterns
❌ Don't:

Make it too rigid with MUST/ALWAYS
Over-specify edge cases
Rely on exact keyword matching
Include malware or harmful content
✅ Do:

Explain the reasoning
Allow flexibility
Show genuine expertise
Help users think better