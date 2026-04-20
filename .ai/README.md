# AI Agents Directory

Unified location for AI agent configurations and context.

## Active Agents

| Agent | Location | Status |
|-------|----------|--------|
| Windsurf | `.windsurf/` | ✅ Active - Primary IDE |
| Claude | `.claude/` | ✅ Active - Context preserved |

## Structure

```
.ai/
├── README.md          # This file
├── agents/            # Future: shared agent configs
└── context/           # Future: shared context across agents

.windsurf/            # Windsurf-specific (separate)
├── rules/
├── skills/
└── workflows/

.claude/              # Claude-specific context (preserved)
```

## Agent Integration

Each agent has its own subdirectory with:
- `rules/` - Constraints and guidelines
- `skills/` - Specialized knowledge
- `workflows/` - Step-by-step procedures
- `context/` - Session history and memory

## Adding New Agents

1. Create `.{agent-name}/` directory
2. Add to table above
3. Link shared resources from `.ai/` if needed
