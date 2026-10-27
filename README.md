```{=html}
<p align="center">
```
`<img src="assets/kryx-logo.png" alt="KRYX — Autonomous Coding Agent" width="360">`{=html}
```{=html}
</p>
```
```{=html}
<h1 align="center">
```
KRYX
```{=html}
</h1>
```
```{=html}
<p align="center">
```
`<strong>`{=html}Sense the code. Shape the
outcome.`</strong>`{=html}`<br>`{=html} An autonomous coding agent for
your terminal.
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<a href="https://pypi.org/project/kryx/">`{=html}`<img src="https://img.shields.io/pypi/v/kryx?label=PyPI" alt="PyPI version">`{=html}`</a>`{=html}
`<a href="https://pypi.org/project/kryx/">`{=html}`<img src="https://img.shields.io/pypi/pyversions/kryx" alt="Python versions">`{=html}`</a>`{=html}
`<a href="LICENSE">`{=html}`<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">`{=html}`</a>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Overview

**KRYX** is a terminal-based AI coding agent designed to bring
AI-assisted development into your command line. Choose a supported model
provider, configure your model, and give KRYX a prompt from your
terminal.

## Features

-   **Terminal-first workflow** --- work from your command line.
-   **Multiple provider options** --- OpenAI, OpenRouter, Gemini/Google,
    and Anthropic.
-   **Model selection** --- specify the model you want to use.
-   **Prompt from the command line** --- pass a task directly as an
    argument.
-   **Optional save flag** --- use `--save` when you want to save a run
    or result, according to the CLI's supported behavior.

## Requirements

-   Python **3.12 or newer**
-   An API key for the provider you choose

## Installation

Install KRYX from PyPI:

``` bash
python -m pip install kryx
```

Check that the command is available:

``` bash
kryx --help
```

> If your machine has multiple Python versions installed, make sure
> `pip` belongs to Python 3.12 or newer. You can also install with
> `python3.12 -m pip install kryx` where that command is available.

## Quick start

See the available options:

``` bash
kryx --help
```

A basic prompt:

``` bash
kryx "Explain the structure of this project"
```

Select a provider and model:

``` bash
kryx --provider openai --model MODEL_NAME "Review this code"
```

KRYX's CLI supports these provider choices:

-   `openai`
-   `openrouter`
-   `gemini`
-   `google`
-   `anthropic`

Replace `MODEL_NAME` with a model supported by your selected provider.

### API keys

Pass a key with the CLI option when needed:

``` bash
kryx --provider openai --api-key YOUR_API_KEY --model MODEL_NAME "Help me understand this function"
```

**Security note:** Avoid putting real API keys directly into shell
history, screenshots, logs, or source control. Prefer a secure
environment variable or configuration mechanism if supported by your
setup.

## Command-line options

  -----------------------------------------------------------------------
  Option                              Description
  ----------------------------------- -----------------------------------
  `--help`                            Show the help message and exit.

  `--provider`                        Choose a provider: `openai`,
                                      `openrouter`, `gemini`, `google`,
                                      or `anthropic`.

  `--model`                           Select the model to use.

  `--api-key`                         Provide the provider API key.

  `--save`                            Enable the CLI's save option.

  `prompt`                            The task or question to send to
                                      KRYX.
  -----------------------------------------------------------------------

Run `kryx --help` for the options available in your installed version.

## Development

Clone the repository and enter the project directory:

``` bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
```

Create and activate a virtual environment, then install the project in
editable mode:

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS/Linux:

``` bash
source .venv/bin/activate
python -m pip install -e .
```

Build a release distribution:

``` bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The generated `dist/` and `build/` directories are build artifacts; they
generally do not need to be committed to Git.

## Contributing

Bug reports, ideas, and pull requests are welcome. When reporting an
issue, include your Python version, operating system, the command you
ran, and the relevant error output. **Never include API keys, access
tokens, or other secrets.**


------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<sub>`{=html} KRYX · Autonomous Coding Agent `</sub>`{=html}
```{=html}
</p>
```
