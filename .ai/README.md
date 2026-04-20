# AI Agents Directory

Unified location for AI agent configurations and context.

Start with `WORKFLOW_CONTROL_PLANE.md` for task routing, prompt shape, validation gates, and
safety rules. Agent-specific folders still hold their native rules, skills, workflows, and memory.

## Active Agents

| Agent | Location | Status |
|---|---|---|
| Windsurf | `.windsurf/` | Active - primary IDE |
| Claude | `.claude/` | Active - context preserved |
| Codex | `.codex/` | Active - local skills |
| Gemini | `.gemini/` | Active - validation/audit context |

## Structure

```text
.ai/
├── README.md
├── WORKFLOW_CONTROL_PLANE.md
├── agents/
└── context/

.windsurf/
├── rules/
├── skills/
└── workflows/

.claude/
├── agents/
├── commands/
└── skills/
```

## Agent Integration

Each agent can keep its own conventions:

- `rules/` - constraints and guidelines.
- `skills/` - specialized knowledge.
- `workflows/` - step-by-step procedures.
- `context/` - session history and memory.

The shared control plane defines when to use those resources and which validation gates are required
before handoff or promotion.

## Adding New Agents

1. Create `.{agent-name}/` or add the agent under the relevant platform folder.
2. Add the agent to the active table above.
3. Link shared resources from `.ai/` when possible.
4. Add any task-specific routing rule to `WORKFLOW_CONTROL_PLANE.md`.
