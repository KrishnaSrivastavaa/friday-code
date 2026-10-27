![KRYX — Autonomous Coding Agent](assets/kryx-logo.png)

# KRYX

**Sense the code. Shape the outcome.**

An autonomous coding agent for your terminal.

[![PyPI version](https://img.shields.io/pypi/v/kryx?label=PyPI)](https://pypi.org/project/kryx/)
[![Python versions](https://img.shields.io/pypi/pyversions/kryx)](https://pypi.org/project/kryx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**KRYX** brings AI-assisted coding into your terminal. Choose a supported model provider, select a model, and send a prompt directly from the command line.

## Features

- **Terminal-first workflow** — work from your command line.
- **Multiple provider options** — OpenAI, OpenRouter, Gemini/Google, and Anthropic.
- **Model selection** — specify which model to use.
- **Direct prompts** — pass a task or question as a command-line argument.
- **Optional save flag** — use `--save` for the CLI's save functionality.

## Requirements

- Python **3.12 or newer**
- An API key for the provider you choose

## Installation

Install KRYX from PyPI:

```bash
python -m pip install kryx
```

Check the available options:

```bash
kryx --help
```

> If you have multiple Python versions installed, make sure `pip` belongs to Python 3.12 or newer.

## Quick start

Ask KRYX a question:

```bash
kryx "Explain the structure of this project"
```

Choose a provider and model:

```bash
kryx --provider openai --model MODEL_NAME "Review this code"
```

Supported provider choices:

- `openai`
- `openrouter`
- `gemini`
- `google`
- `anthropic`

Replace `MODEL_NAME` with a model supported by your selected provider.

### API keys

You can pass an API key using the CLI option:

```bash
kryx --provider openai --api-key YOUR_API_KEY --model MODEL_NAME "Explain this function"
```

**Security note:** Avoid putting real API keys in shell history, screenshots, logs, or source control. Never commit API keys or `.env` files.

## Command-line options

| Option | Description |
| --- | --- |
| `--help` | Show the help message and exit. |
| `--provider` | Choose a provider: `openai`, `openrouter`, `gemini`, `google`, or `anthropic`. |
| `--model` | Select the model to use. |
| `--api-key` | Provide the provider API key. |
| `--save` | Enable the CLI's save option. |
| `prompt` | The task or question to send to KRYX. |

Run `kryx --help` to see the options available in your installed version.

## Development

Clone your repository and enter the project directory:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
```

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Build and validate a release:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The `dist/` and `build/` directories contain generated artifacts and generally should not be committed to Git.

## Contributing

Bug reports, feature ideas, and pull requests are welcome. When reporting an issue, include your Python version, operating system, the command you ran, and the relevant error output. Do not include secrets or API keys.



---

*KRYX · Autonomous Coding Agent*
