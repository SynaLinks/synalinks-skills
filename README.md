# Claude Skills for Synalinks

---

This repository contains skills to use with Claude Code.

## What Are Claude Skills?

Claude Skills are customizable workflows that teach Claude how to perform specific tasks according to your unique requirements. Skills enable Claude to execute tasks in a repeatable, standardized manner across all Claude platforms.

## What is the goal of Synalinks skills?

The goal of Synalinks skills is to teach Claude to use the Synalinks framework correctly. Synalinks is Keras-inspired, so without guidance LMs tend to mix Keras / LangChain / DSPy syntax — producing plausible-looking but broken code. These skills constrain Claude to idiomatic Synalinks usage.

## The `synalinks` skill

This repo ships a **single skill**, `synalinks`, that covers the whole
framework. Claude auto-activates it based on its `SKILL.md` `description`. The
`SKILL.md` is a scannable overview, organized into one section per slice of the
framework, each pointing to a runnable script (and its captured run log) and a
deep-dive reference doc:

| Section | What it covers |
|---------|----------------|
| **Core** | DataModel, Field, Input, JSON operators (`+ & \| ^ ~`), `synalinks.ops`, configuration, LanguageModel/EmbeddingModel basics |
| **Programs** | Program class, four building APIs (Functional / Sequential / Subclassing / Mixed), multi-input/output graphs, build/call lifecycle, save/load, `summary`, `get_module`, `plot_program`, custom serialization |
| **Modules** | Generator, ChainOfThought, SelfCritique, Identity, PythonSynthesis, SequentialPlanSynthesis, custom modules via subclassing |
| **Control flow** | Decision, Branch, parallel branches, self-consistency, XOR input/output guards, And/Or modules, branch merging |
| **Agents** | FunctionCallingAgent, RLM, DeepAgent, Tool definitions, MCP integration (MultiServerMCPClient), subagents, trajectories |
| **Knowledge** | KnowledgeBase (DuckDB), EmbedKnowledge, UpdateKnowledge, RetrieveKnowledge, RAG/KAG, hybrid search, Entity/Relation graphs |
| **Training** | `compile()` / `fit()` / `evaluate()` / `predict()`, callbacks, ProgramCheckpoint, training workflow |
| **Rewards** | ExactMatch, CosineSimilarity, LMAsJudge, ProgramAsJudge, F1Score, custom rewards/metrics, masking |
| **Optimizers** | RandomFewShot, OMEGA, Dominated Novelty Search, mutation/crossover, quality-diversity tuning |
| **Providers** | Provider prefixes (openai, anthropic, groq, openrouter, cohere, deepseek, together_ai, bedrock, doubleword, hosted_vllm, ...), local OpenAI-compatible servers (LMStudio/vLLM/ollama), OpenAI-compatible setups (e.g. OpenRouter embeddings via `api_base`) |
| **Datasets** | Built-in datasets (gsm8k, hotpotqa, arcagi), custom iterable datasets, visualization (`plot_program`, `plot_history`, `plot_metrics_*`) |

The skill folder contains:
- `SKILL.md` — frontmatter + scannable usage guide for the whole framework
- `scripts/` — runnable example scripts (each docstring states its `Usage:` and `Run log:`)
- `references/` — deep-dive reference docs plus the captured run logs (`*.log`)

## Install

The recommended way to install this skill is the official
[`skills`](https://skills.sh) CLI (`npx skills`). It clones this repo, discovers
the `synalinks` skill, and wires it into your agent's skills directory —
no manual cloning, copying, or zipping required. It needs Node.js (the `npx`
that ships with it); nothing else.

### Quick start (Claude Code)

Install the skill, globally (available in all your projects):

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks --global
```

Or install into the **current project only** (commit `.claude/skills/` so
teammates pick it up automatically) — run from your project root:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks
```

The CLI auto-detects Claude Code and installs to `~/.claude/skills/` (with
`--global`) or `<repo>/.claude/skills/` (project scope, the default). By default
it symlinks the skill; pass `--copy` to copy the files instead.

List what's in the repo without installing anything:

```shell
npx skills add SynaLinks/synalinks-skills --list
```

Run `npx skills add SynaLinks/synalinks-skills` with no flags for an
interactive picker.

### Other agents

The `skills` CLI supports 70+ agents. Target one (or all) explicitly with
`-a`/`--agent`:

```shell
npx skills add SynaLinks/synalinks-skills --skill synalinks -a claude-code   # one agent
npx skills add SynaLinks/synalinks-skills --all                              # everything, all agents
```

### Verify

```shell
npx skills list            # list installed skills
```

In Claude Code you can also run the `/skills` slash command — the `synalinks`
skill should appear. Claude auto-activates it for your task based on its
`SKILL.md` `description` field — no manual selection needed.

### Claude.ai (web) and Claude Desktop

Web/Desktop uploads expect a single `.skill` archive (a zip of the skill folder
with `SKILL.md` at the root). Clone the repo and build the archive:

```shell
git clone https://github.com/SynaLinks/synalinks-skills.git
cd synalinks-skills/skills
zip -r ../synalinks.skill synalinks
```

Then in the Claude interface, click the skill icon (🧩), choose "Upload
skill", and upload the `.skill` file. See [Using skills with Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude#h_c6008b84ad)
for the latest UI walkthrough.

## Updating

```shell
npx skills update          # update all installed skills to the latest version
```

Use `-g`/`--global` or `-p`/`--project` to scope the update. For
Claude.ai / Desktop, regenerate the `.skill` archive and re-upload it.

## Uninstall

```shell
npx skills remove synalinks
```

Add `-g`/`--global` to remove a globally-installed skill. For Claude.ai /
Desktop, remove the skill from the skill icon (🧩) menu.

# License

These skills are licensed under Apache 2.0, like Synalinks framework.

See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknoledgement

These skills have been created by [Ramiro Salas](https://www.linkedin.com/in/rsalas/) the CTO of [Hexatropian](https://www.linkedin.com/company/hextropian-systems/) an active early member of [Synalinks](https://github.com/SynaLinks/synalinks) community.

## Synalinks Project

- [Synalinks Framework](https://github.com/SynaLinks/synalinks) - The neuro-symbolic AI framework these skills are designed for
