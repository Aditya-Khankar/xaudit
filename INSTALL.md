# Installation

## Requirements

- Python 3.11 or higher
- pip

Check your Python version:
```bash
python --version
```

If below 3.11, install from https://python.org/downloads

## Standard install (Analysis only — no API key needed)

```bash
git clone https://github.com/Aditya-Khankar/cognidrift
cd cognidrift
pip install -e .
cognidrift demo
```

That's it. The demo runs immediately with no configuration.

## Install with trace generation (Optional)

Only needed if you want to generate synthetic agent traces using Gemini.

```bash
pip install -e ".[generate]"
```

Get a key at **[Google AI Studio](https://aistudio.google.com/apikey)**.

Then set up your API key:

```bash
cp .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual key.

### Troubleshooting the generator
If you see a quota or permission error:
*   **Key Type**: Create your key in **Google AI Studio**, not just a generic Google Cloud API key.
*   **API Enabled**: Verify that the **"Generative Language API"** is enabled for your project.
*   **Tiers**: Free-tier keys have rate limits. If you've generated many traces recently, wait 60 seconds and try again.

## Windows-specific notes

- Use `python` not `python3`
- Use Command Prompt or PowerShell — Git Bash may have path issues
- If the `cognidrift` command is not found after install, run: `python -m cognidrift`

## Troubleshooting?

**`pip install -e .` fails with "No module named setuptools"**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

**Cognidrift command not found after installation**
```bash
python -m pip install -e .
python -m cognidrift demo
```

**`ModuleNotFoundError` on any import**
```bash
pip install -r requirements.txt
pip install -e .
```

**On Windows: encoding errors in output**
```bash
set PYTHONIOENCODING=utf-8
cognidrift demo
```
