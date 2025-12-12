# References and Third-Party Integrations

**Last Updated**: December 12, 2025  
**Purpose**: Attribution and acknowledgment of upstream projects and libraries

---

This document lists the third-party libraries, tools, and upstream projects that AI-OS depends on or is inspired by. We are grateful to the open-source community for making these tools available.

---

## Core Dependencies

### Machine Learning & Deep Learning

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **PyTorch** | BSD-3-Clause | Deep learning framework | [Website](https://pytorch.org/) · [GitHub](https://github.com/pytorch/pytorch) |
| **Transformers** | Apache-2.0 | State-of-the-art NLP models | [Website](https://huggingface.co/transformers/) · [GitHub](https://github.com/huggingface/transformers) |
| **Accelerate** | Apache-2.0 | Distributed training utilities | [GitHub](https://github.com/huggingface/accelerate) |
| **Datasets** | Apache-2.0 | Dataset loading and processing | [GitHub](https://github.com/huggingface/datasets) |
| **PEFT** | Apache-2.0 | Parameter-efficient fine-tuning | [GitHub](https://github.com/huggingface/peft) |
| **Safetensors** | Apache-2.0 | Safe tensor serialization | [GitHub](https://github.com/huggingface/safetensors) |
| **Hugging Face Hub** | Apache-2.0 | Model hub integration | [GitHub](https://github.com/huggingface/huggingface_hub) |

### Memory Optimization & Performance

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **DeepSpeed** | Apache-2.0 | Deep learning optimization library | [Website](https://www.deepspeed.ai/) · [GitHub](https://github.com/microsoft/DeepSpeed) |
| **bitsandbytes** | MIT | 8-bit optimizers and quantization | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| **FlashAttention** | BSD-3-Clause | Fast and memory-efficient attention | [GitHub](https://github.com/Dao-AILab/flash-attention) |

### Tokenization

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **SentencePiece** | Apache-2.0 | Unsupervised text tokenizer | [GitHub](https://github.com/google/sentencepiece) |
| **Protobuf** | BSD-3-Clause | Protocol buffers for tokenizers | [GitHub](https://github.com/protocolbuffers/protobuf) |

### Evaluation

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **lm-eval** | MIT | Language model evaluation harness | [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) |
| **math-verify** | MIT | Mathematical verification utilities | [PyPI](https://pypi.org/project/math-verify/) |

---

## Application Dependencies

### Web & Networking

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **aiohttp** | Apache-2.0 | Async HTTP client/server | [GitHub](https://github.com/aio-libs/aiohttp) |
| **httpx** | BSD-3-Clause | Modern HTTP client | [GitHub](https://github.com/encode/httpx) |
| **Playwright** | Apache-2.0 | Browser automation | [GitHub](https://github.com/microsoft/playwright-python) |
| **BeautifulSoup4** | MIT | HTML/XML parsing | [Website](https://www.crummy.com/software/BeautifulSoup/) |
| **lxml** | BSD-3-Clause | XML and HTML processing | [GitHub](https://github.com/lxml/lxml) |
| **Trafilatura** | Apache-2.0 | Web scraping and text extraction | [GitHub](https://github.com/adbar/trafilatura) |

### Data & Validation

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **Pydantic** | MIT | Data validation using Python types | [GitHub](https://github.com/pydantic/pydantic) |
| **orjson** | Apache-2.0 / MIT | Fast JSON library | [GitHub](https://github.com/ijl/orjson) |
| **PyYAML** | MIT | YAML parser and emitter | [GitHub](https://github.com/yaml/pyyaml) |
| **NumPy** | BSD-3-Clause | Numerical computing | [Website](https://numpy.org/) · [GitHub](https://github.com/numpy/numpy) |

### CLI & UI

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **Typer** | MIT | CLI application framework | [GitHub](https://github.com/tiangolo/typer) |
| **Rich** | MIT | Rich text and formatting | [GitHub](https://github.com/Textualize/rich) |
| **Pillow** | HPND | Python Imaging Library | [GitHub](https://github.com/python-pillow/Pillow) |
| **pystray** | LGPL-3.0 | System tray icon library | [GitHub](https://github.com/moses-palmer/pystray) |
| **matplotlib** | PSF | Plotting library | [GitHub](https://github.com/matplotlib/matplotlib) |
| **Markdown** | BSD-3-Clause | Markdown parser | [GitHub](https://github.com/Python-Markdown/markdown) |
| **tkinterweb** | MIT | Web browser widget for Tkinter | [GitHub](https://github.com/AKEric/tkinterweb) |
| **rapidfuzz** | MIT | Fast string matching | [GitHub](https://github.com/maxbachmann/RapidFuzz) |

### System & Utilities

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **psutil** | BSD-3-Clause | Process and system utilities | [GitHub](https://github.com/giampaolo/psutil) |
| **watchdog** | Apache-2.0 | Filesystem event monitoring | [GitHub](https://github.com/gorakhargosh/watchdog) |
| **tenacity** | Apache-2.0 | Retry library | [GitHub](https://github.com/jd/tenacity) |
| **tqdm** | MIT / MPL-2.0 | Progress bars | [GitHub](https://github.com/tqdm/tqdm) |
| **dbus-next** | MIT | D-Bus for Python (Linux) | [GitHub](https://github.com/altdesktop/python-dbus-next) |

---

## Development Dependencies

| Library | License | Description | Links |
|---------|---------|-------------|-------|
| **pytest** | MIT | Testing framework | [GitHub](https://github.com/pytest-dev/pytest) |
| **pytest-asyncio** | Apache-2.0 | Async test support | [GitHub](https://github.com/pytest-dev/pytest-asyncio) |
| **Ruff** | MIT | Fast Python linter | [GitHub](https://github.com/astral-sh/ruff) |
| **Black** | MIT | Code formatter | [GitHub](https://github.com/psf/black) |

---

## Documentation

| Tool | License | Description | Links |
|------|---------|-------------|-------|
| **MkDocs** | BSD-2-Clause | Static site generator | [GitHub](https://github.com/mkdocs/mkdocs) |
| **Material for MkDocs** | MIT | MkDocs theme | [GitHub](https://github.com/squidfunk/mkdocs-material) |

---

## Research References

AI-OS implements concepts from the following research:

### Hierarchical Reasoning Models (HRM)
- **Paper**: [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- Implements hierarchical reasoning with mixture-of-experts architecture

### Mixture of Experts (MoE)
- **Switch Transformers**: [Paper](https://arxiv.org/abs/2101.03961) - Scaling to trillion parameter models
- **GShard**: [Paper](https://arxiv.org/abs/2006.16668) - Scaling giant models with conditional computation

### Memory Optimization Techniques
- **Gradient Checkpointing**: Trade compute for memory during backpropagation
- **Mixed Precision Training**: FP16/BF16 training with loss scaling
- **ZeRO Optimization**: Memory-efficient distributed training (DeepSpeed)

---

## Acknowledgments

Special thanks to:
- The **PyTorch** team for the foundational deep learning framework
- **Hugging Face** for the transformers ecosystem and model hub
- **Microsoft DeepSpeed** team for training optimizations
- The **EleutherAI** team for the evaluation harness
- All contributors to the open-source ML community

---

## License Compliance

All dependencies are used in compliance with their respective licenses. AI-OS is licensed under the **AI-OS Non‑Selling Attribution License (ANSL) v1.0** - see [LICENSE](LICENSE) for details.

For questions about licensing or attribution, please open an issue on [GitHub](https://github.com/Wulfic/AI-OS/issues).

---

[Back to README](README.md) | [Back to Documentation](docs/README.md)
