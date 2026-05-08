# Claude Skills for Synalinks

---

This repository contains skills to use with Claude Code.

## What Are Claude Skills?

Claude Skills are customizable workflows that teach Claude how to perform specific tasks according to your unique requirements. Skills enable Claude to execute tasks in a repeatable, standardized manner across all Claude platforms.

## What is the goal of Synalinks skills?

The goal of Synalinks skills is to teach Claude to use the Synalinks framework correctly. Synalinks is Keras-inspired, so without guidance LMs tend to mix Keras / LangChain / DSPy syntax — producing plausible-looking but broken code. These skills constrain Claude to idiomatic Synalinks usage.

## The 11 Skills

Each skill targets one slice of the framework. Claude auto-activates the relevant skill(s) based on what you're asking about.

| Skill | When it activates |
|-------|-------------------|
| **synalinks-core** | DataModel, Field, Input, JSON operators (`+ & \| ^ ~`), `synalinks.ops`, configuration, LanguageModel/EmbeddingModel basics |
| **synalinks-programs** | Program class, four building APIs (Functional / Sequential / Subclassing / Mixed), multi-input/output graphs, build/call lifecycle, save/load, `summary`, `get_module`, `plot_program`, custom serialization |
| **synalinks-modules** | Generator, ChainOfThought, SelfCritique, Identity, PythonSynthesis, SequentialPlanSynthesis, custom modules via subclassing |
| **synalinks-control-flow** | Decision, Branch, parallel branches, self-consistency, XOR input/output guards, And/Or modules, branch merging |
| **synalinks-agents** | FunctionCallingAgent, ToolCalling, Tool definitions, MCP integration (MultiServerMCPClient), trajectories |
| **synalinks-knowledge** | KnowledgeBase (DuckDB), EmbedKnowledge, UpdateKnowledge, RetrieveKnowledge, RAG/KAG, hybrid search, Entity/Relation graphs |
| **synalinks-training** | `compile()` / `fit()` / `evaluate()` / `predict()`, callbacks, ProgramCheckpoint, training workflow |
| **synalinks-rewards** | ExactMatch, CosineSimilarity, LMAsJudge, ProgramAsJudge, MeanRewardWrapper, F1Score, custom rewards/metrics, masking |
| **synalinks-optimizers** | RandomFewShot, OMEGA, Dominated Novelty Search, mutation/crossover, quality-diversity tuning |
| **synalinks-providers** | Provider prefixes (openai, anthropic, groq, openrouter, cohere, deepseek, together_ai, bedrock, doubleword, hosted_vllm, ...), local servers (LMStudio/vLLM), OpenRouter embeddings |
| **synalinks-datasets** | Built-in datasets (gsm8k, hotpotqa, arcagi), custom iterable datasets, visualization (`plot_program`, `plot_history`, `plot_metrics_*`) |

Each skill folder contains:
- `SKILL.md` — frontmatter + scannable usage guide
- `references/` — deep-dive reference docs
- `scripts/` — runnable example scripts

## Install

Clone the repository — every install path below copies, zips, or symlinks the
`synalinks-*/` folders out of this checkout, so keep it somewhere stable.

```shell
git clone https://github.com/SynaLinks/synalinks-skills.git
cd synalinks-skills
```

Pick one of the install targets below depending on where you use Claude.

### Claude Code

Skills live in one of two locations:

| Scope | Path | When to use |
|-------|------|-------------|
| **User** (all projects) | `~/.claude/skills/` | You always want the skills available, regardless of which repo you're in. |
| **Project** (one repo)  | `<repo>/.claude/skills/` | Limit the skills to a single Synalinks project; commit them with the repo so teammates pick them up automatically. |

Install at the user level (recommended for solo use):

```shell
mkdir -p ~/.claude/skills/
cp -r synalinks-* ~/.claude/skills/
```

Or install for a single project:

```shell
mkdir -p /path/to/your/synalinks-project/.claude/skills/
cp -r synalinks-* /path/to/your/synalinks-project/.claude/skills/
```

> **Tip:** prefer `ln -s "$PWD"/synalinks-* ~/.claude/skills/` if you want
> updates from `git pull` to flow through automatically.

Verify the install — start Claude Code and run the `/skills` slash command;
the 11 `synalinks-*` skills should appear in the list. You can also peek at a
skill's frontmatter directly:

```shell
head ~/.claude/skills/synalinks-core/SKILL.md
```

Claude auto-activates the relevant skill(s) for your task based on each
`SKILL.md`'s `description` field — no manual selection needed.

### Claude.ai (web) and Claude Desktop

These uploads expect a single `.skill` archive (a zip of one skill folder
with `SKILL.md` at the root). Build one archive per skill:

```shell
for d in synalinks-*/; do zip -r "${d%/}.skill" "$d"; done
```

Then in the Claude interface, click the skill icon (🧩), choose "Upload
skill", and upload each `.skill` file. Repeat for every skill you want
available. See [Using skills with Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude#h_c6008b84ad)
for the latest UI walkthrough.

## Updating

```shell
cd synalinks-skills
git pull
cp -r synalinks-* ~/.claude/skills/    # if you copied (skip if you symlinked)
```

For Claude.ai / Desktop, regenerate the `.skill` archives and re-upload the
ones that changed. Restart Claude Code (or run `/skills` → reload) so the
new metadata is picked up.

## Uninstall

```shell
rm -rf ~/.claude/skills/synalinks-*
```

For Claude.ai / Desktop, remove each skill from the skill icon (🧩) menu.

# License

These skills are licensed under Apache 2.0, like Synalinks framework.

See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknoledgement

These skills have been created by [Ramiro Salas](https://www.linkedin.com/in/rsalas/) the CTO of [Hexatropian](https://www.linkedin.com/company/hextropian-systems/) an active early member of [Synalinks](https://github.com/SynaLinks/synalinks) community.

## Synalinks Project

- [Synalinks Framework](https://github.com/SynaLinks/synalinks) - The neuro-symbolic AI framework these skills are designed for
