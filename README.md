# Claude Skills for Synalinks

---

This repository contains skills to use with Claude Code.

## What Are Claude Skills?

Claude Skills are customizable workflows that teach Claude how to perform specific tasks according to your unique requirements. Skills enable Claude to execute tasks in a repeatable, standardized manner across all Claude platforms.

## What is the goal of Synalinks skills?

The goal of Synalinks skills is to teach Claude to use the Synalinks framework correctly. Synalinks is Keras-inspired, so without guidance LMs tend to mix Keras / LangChain / DSPy syntax — producing plausible-looking but broken code. These skills constrain Claude to idiomatic Synalinks usage.

## The 10 Skills

Each skill targets one slice of the framework. Claude auto-activates the relevant skill(s) based on what you're asking about.

| Skill | When it activates |
|-------|-------------------|
| **synalinks-core** | DataModel, Field, Program (functional/sequential/subclassing), JSON operators (`+ & \| ^ ~`), saving/loading, configuration, LanguageModel/EmbeddingModel basics |
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

```shell
git clone https://github.com/SynaLinks/synalinks-skills.git
cd synalinks-skills
```

## Using Skills in Claude

Click the skill icon (🧩) in your chat interface, then add skills from the marketplace or upload your own. Claude automatically activates the relevant skill(s) for your task.

To use these skills with Claude API and Claude.ai, zip each skill folder:

```shell
for d in synalinks-*/; do zip -r "${d%/}.skill" "$d"; done
```

See [Using skills with Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude#h_c6008b84ad) for more information.

### Using Skills in Claude Code

Place the skill folders in `~/.config/claude-code/skills/`:

```shell
mkdir -p ~/.config/claude-code/skills/
cp -r synalinks-* ~/.config/claude-code/skills/
```

Verify a skill's metadata:

```shell
head ~/.config/claude-code/skills/synalinks-core/SKILL.md
```

Start Claude Code:

```shell
claude
```

Claude loads all 10 skills and auto-activates the relevant ones based on your task.

# License

These skills are licensed under Apache 2.0, like Synalinks framework.

See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknoledgement

These skills have been created by [Ramiro Salas](https://www.linkedin.com/in/rsalas/) the CTO of [Hexatropian](https://www.linkedin.com/company/hextropian-systems/) an active early member of [Synalinks](https://github.com/SynaLinks/synalinks) community.

## Synalinks Project

- [Synalinks Framework](https://github.com/SynaLinks/synalinks) - The neuro-symbolic AI framework these skills are designed for
