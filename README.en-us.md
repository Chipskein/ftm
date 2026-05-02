# 🈯 FTM — Manga Translation Tool

A modular manga translation pipeline with GPU acceleration that automatically detects speech bubbles, extracts Japanese text, translates it, and inserts the result back into the image.

---
## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
  - [Via pip (recommended)](#via-pip-recommended)
  - [Via local clone](#via-local-clone)
  - [Translation model (Ollama)](#translation-model-ollama)
- [🛠️ CLI Reference](#️-cli-reference)
  - [Required](#required)
  - [Optional](#optional)
- [🔧 Examples](#-examples)
- [🗂️ Output Files](#️-output-files)
- [🧩 Pipeline Steps](#-pipeline-steps)
- [📋 `bubbles.json` Format](#-bubblesjson-format)
- [🤝 Contributing](#-contributing)
- [⚠️ Legal Disclaimer](#️-legal-disclaimer)

---

## ✨ Features

- **Bubble Detection** — YOLO-based detection
- **Text Extraction** — MangaOCR for accurate Japanese recognition
- **Translation** — Ollama models (`translategemma:4b` / `translategemma:12b`)
- **Typesetting** — Renders translated text back into the bubbles
- **Modular Pipeline** — Run all steps or only the ones you need
- **CPU/GPU Control** — Granular per-step configuration
- **Resource Monitoring** — Optional CPU/RAM profiling saved to CSV

---

## 🚀 Quick Start

```bash
python -m ftm --image ./manga_page.jpg
```

Runs the full pipeline (detection → extraction → translation → typesetting) and saves the translated image to `./tmp/`.

---

## 📦 Installation

> **Requirements:** Python 3.9+, [Ollama](https://ollama.com/) running locally, and optionally a CUDA-compatible GPU.

### Via pip (recommended)

**Standard installation:**
```bash
pip install git+https://github.com/Chipskein/ftm.git
```

**With NVIDIA GPU support (nvidia-ml-py):**
```bash
pip install "ftm[nvidia] @ git+https://github.com/Chipskein/ftm.git"
```

**Specific version or branch:**
```bash
# specific branch
pip install git+https://github.com/Chipskein/ftm.git@main

# specific tag/version
pip install git+https://github.com/Chipskein/ftm.git@v0.0.1
```

### Via local clone

```bash
git clone https://github.com/Chipskein/ftm.git
cd ftm

# standard installation
pip install .

# with NVIDIA support
pip install ".[nvidia]"
```

### Translation model (Ollama)

After installing, download the translation model via Ollama:

```bash
ollama pull translategemma:4b

# or the larger model for better quality
ollama pull translategemma:12b
```

---

## 🛠️ CLI Reference

```
python -m ftm --image <path> [options]
```

### Required

| Argument | Description |
|---|---|
| `--image PATH` | Path to the input image |

### Optional

| Argument | Default | Description |
|---|---|---|
| `--output PATH` | `tmp/<name>_translated.jpg` | Path for the translated output image |
| `--tmp DIR` | `./tmp` | Directory for temporary files and intermediate results |
| `--debug` | `False` | Enables detailed logging and saves all crops and debug images |
| `--steps` | all | Steps to run: `detection` `extraction` `translation` `typesetting` |
| `--bubbles-json PATH` | `None` | Loads bubble data from a JSON file, skipping detection |
| `--translate-lang` | `en` | Target language: `en` or `pt` |
| `--translate-model` | `translategemma:4b` | Ollama model: `translategemma:4b` or `translategemma:12b` |
| `--ollama-host URL` | `http://localhost:11434` | Custom Ollama API host |
| `--ollama-model-temperature` | `0.0` | Model temperature (0.0 = deterministic) |
| `--use-cpu-all` | `False` | Forces CPU usage across all steps |
| `--use-cpu-step` | `[]` | Forces CPU on specific steps: `detection` `extraction` |
| `--use-monitor` | `False` | Saves CPU/RAM usage to CSV in the tmp directory (requires `psutil`) |
| `--huggingface_online_check` | `False` | Checks Hugging Face model availability at startup |

---

## 🔧 Examples

**Translate to Portuguese:**
```bash
python -m ftm --image page.jpg --translate-lang pt
```

**Run only translation and typesetting (using a previously saved JSON):**
```bash
python -m ftm --image page.jpg --bubbles-json ./tmp/page_bubbles.json --steps translation typesetting
```

**Use the larger model for better quality:**
```bash
python -m ftm --image page.jpg --translate-model translategemma:12b
```

**Run on CPU only (no GPU):**
```bash
python -m ftm --image page.jpg --use-cpu-all
```

**Debug mode with resource monitoring:**
```bash
python -m ftm --image page.jpg --debug --use-monitor
```

**Custom Ollama host (e.g. remote server):**
```bash
python -m ftm --image page.jpg --ollama-host http://192.168.1.100:11434
```

---

## 🗂️ Output Files

After a successful run, the following files are saved to the `--tmp` directory:

| File | Description |
|---|---|
| `<name>_translated.jpg` | Final translated image |
| `<name>_bubbles.json` | Bubble metadata: coordinates, OCR text, translations, and timings |
| `resources_<name>_yolo.csv` | Resource usage log (if `--use-monitor` is active) |

---

## 🧩 Pipeline Steps

```
Input Image
       │
       ▼
┌─────────────┐
│  Detection  │  YOLO detects bubble regions → crops saved to tmp/
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extraction │  MangaOCR reads Japanese text from each crop
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Translation │  Ollama LLM translates JP → target language
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Typesetting │  Translated text rendered back onto the original image
└─────────────┘
       │
       ▼
Output Image + bubbles.json
```

Each step can be run independently using `--steps` and `--bubbles-json`.

---

## 📋 `bubbles.json` Format

The intermediate JSON file stores data for each detected bubble:

```json
[
  {
    "crop": "./tmp/page_crop_0.jpg",
    "jp_text": "こんにちは！",
    "translated_text": "Hello!",
    "extraction_symbols": 6,
    "extraction_time_s": 0.42,
    "translation_symbols": 6,
    "translating_time_s": 1.13
  }
]
```

---

## 🤝 Contributing

Pull requests are welcome! To get started:

```bash
git checkout -b feature/my-feature
# make your changes
git commit -m "feat: my feature"
git push origin feature/my-feature
```

For larger changes, please open an issue first for discussion.

---

## ⚠️ Legal Disclaimer

> **PLEASE READ CAREFULLY BEFORE USING THIS TOOL.**

### Purpose

FTM — Manga Translation Tool was developed **exclusively for educational purposes, research, natural language processing study, and personal non-commercial use**. Any use outside this scope is the sole responsibility of the user.

### Prohibition of Use for Scanlations

**It is expressly prohibited** to use this software to produce, reproduce, distribute, publish, or commercialize unauthorized translations of copyrighted works (*scanlations*).

The reproduction and distribution of works without the express authorization of copyright holders may constitute a violation of the following laws, depending on the user's country:

- **Brazil:** Law No. 9,610/1998 (Copyright Law)
- **European Union:** Directive 2019/790 on Copyright in the Digital Single Market
- **United States:** Digital Millennium Copyright Act (DMCA) and 17 U.S.C. § 501
- **Japan:** Copyright Act (著作権法, Law No. 48 of 1970)
- Other applicable national laws and international treaties (Berne Convention, TRIPS)

### Developer Disclaimer

THE DEVELOPERS, CONTRIBUTORS, AND MAINTAINERS OF THIS PROJECT:

1. **Accept no responsibility**, under any circumstances, for any illegal, improper, or harmful use of the software;
2. **Do not endorse, encourage, or support** any use of this tool that violates copyright or any other applicable law;
3. **Provide no warranties** of any kind regarding the software, express or implied, including but not limited to warranties of fitness for a particular purpose;
4. **Are not liable** for any direct, indirect, incidental, special, or consequential damages arising from the use or inability to use the software.

### User Responsibility

**Use of this tool is entirely and exclusively the user's responsibility.** By using FTM, the user represents and warrants that they:

- Are aware of the copyright laws in effect in their country;
- Will use the tool only for legal and authorized purposes;
- Hold the necessary rights over any material they process with the software;
- Release the developers from any civil or criminal liability arising from misuse.

### Respect the Creators

If you enjoy a manga, support the authors and publishers by purchasing official releases or accessing legal reading platforms available in your country.

> **By downloading, cloning, installing, or using this software, you declare that you have read, understood, and fully agreed to this Legal Disclaimer.**