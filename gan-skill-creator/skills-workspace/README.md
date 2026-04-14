# Skills Workspace

This directory contains generated skills and their evaluation data.

## Structure

skills-workspace/ 
├── skill-name-1/ 
│ ├── SKILL.md # Generated skill 
│ ├── iteration-1/ # First test iteration 
│ │ ├── eval-1/ 
│ │ │ ├── with_skill/outputs/ 
│ │ │ ├── without_skill/outputs/ 
│ │ │ └── eval_metadata.json 
│ │ └── benchmark.json 
│ └── iteration-2/ 
│ └── ... (same structure) 
└── skill-name-2/ 
└── ...



## Workflow

1. **Create Skill** → Generate initial SKILL.md
2. **Run Tests** → Execute with-skill and baseline runs
3. **Evaluate** → Review outputs, grade, aggregate
4. **Iterate** → Improve skill, rerun tests
5. **Optimize** → Tune description for triggering

## Files to Track

- `SKILL.md` - The actual skill (keep this!)
- `*/iteration-N/` - Test results (for reference)
- `benchmark.json` - Performance metrics
- `feedback.json` - User feedback from reviews