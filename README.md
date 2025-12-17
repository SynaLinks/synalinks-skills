# Claude Skills for Synalinks

<div align="center">

⭐ If you find Synalinks useful, please star the repo (and [Synalinks](https://github.com/SynaLinks/synalinks) one)! Help us reach more AI/ML engineers and grow the community. ⭐

</div>

---

This repository contains skills to use with Claude Code.

We'll progressively add more skills to cover more advanced Synalinks usecases, so keep tuned!

## What Are Claude Skills?

Claude Skills are customizable workflows that teach Claude how to perform specific tasks according to your unique requirements. Skills enable Claude to execute tasks in a repeatable, standardized manner across all Claude platforms.

## What is the goal of Synalinks skills?

The goal of Synalinks skills is to teach Claude to use properly the Synalinks framework. Given that Synalinks is based on Keras, LMs tend to mixup Keras and other LMs framework syntax witch results in bad practices and buggy code. These skills are provided to streamline the development of neuro-symbolic applications with Synalinks framework.

## Install

```shell
git clone https://github.com/SynaLinks/synalinks-skills.git
cd synalinks-skills
```

## Using Skills in Claude

Click the skill icon (🧩) in your chat interface.
Add skills from the marketplace or upload custom skills.
Claude automatically activates relevant skills based on your task.

To use these skills with Claude API and Claude.ai you'll need to zip the skills with:

```shell
zip -r synalinks.skill synalinks/
```

See [Using skills with Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude#h_c6008b84ad) for more information.

### Using Skills in Claude Code

Place the skill in `~/.config/claude-code/skills/`:

```shell
mkdir -p ~/.config/claude-code/skills/
cp -r synalinks ~/.config/claude-code/skills/
```

Verify skill metadata:

```shell
head ~/.config/claude-code/skills/synalinks/SKILL.md
```

Start Claude Code:

```shell
claude
```

The skill loads automatically and activates when relevant.

# License

These skills are licensed under Apache 2.0, like Synalinks framework.

See the [LICENSE](LICENSE) file for full details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknoledgement

These skills have been created by [Ramiro Salas](https://www.linkedin.com/in/rsalas/) the CTO of [Hexatropian](https://www.linkedin.com/company/hextropian-systems/) an active early member of [Synalinks](https://github.com/SynaLinks/synalinks) community.

## Synalinks Project

- [Synalinks Framework](https://github.com/SynaLinks/synalinks) - The neuro-symbolic AI framework these skills are designed for
