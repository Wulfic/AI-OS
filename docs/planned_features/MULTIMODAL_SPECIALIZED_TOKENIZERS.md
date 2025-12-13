# Multimodal and Specialized Tokenizers

> ## ⚠️ **CRITICAL NOTICE - NOT YET IMPLEMENTED** ⚠️
> 
> **STATUS**: 🚧 **PLANNED FEATURE - NOT CURRENTLY AVAILABLE** 🚧
> 
> The multimodal and specialized tokenizers described in this document are **NOT YET IMPLEMENTED** in AI-OS. This document represents planned future functionality.
> 
> **What works NOW**:
> - ✅ GPT-2, Qwen 2.5, Mistral 7B, Code Llama, DeepSeek-Coder V2, StarCoder2, Phi-3 Mini
> 
> **What does NOT work yet**:
> - ❌ CLIP, LLaVA, SigLIP (Vision/Multimodal)
> - ❌ BioBERT, SciBERT, Legal-BERT, FinBERT (Specialized domains)
> - ❌ Image processing pipeline
> - ❌ Vision-language model support
> 
> **Last Verified**: October 13, 2025
> 
> For currently supported tokenizers, see: `docs/SUPPORTED_TOKENIZERS.md` (to be created)

---

**Date:** October 13, 2025  
**Major Update:** Added Vision, Multimodal, and Domain-Specialized Tokenizers  
**⚠️ Status:** DOCUMENTATION ONLY - IMPLEMENTATION PENDING

---

## 🚀 What's New (PLANNED)

### 🎨 Vision & Multimodal Tokenizers (3)

Train AI models that understand **images, videos, and vision-language tasks**:

| Tokenizer | Vocab | Best For | Downloaded |
|-----------|-------|----------|------------|
| **CLIP (Vision-Language)** 🖼️ | 49,408 | Image-text understanding, zero-shot vision | ✅ Yes |
| **LLaVA 1.5 (Vision Chat)** 💬 | 32,000 | Visual Q&A, image understanding, multimodal chat | ✅ Yes |
| **SigLIP** 🎯 | 32,000 | Advanced vision-language, better than CLIP | ✅ Yes |

### 🔬 Specialized Domain Tokenizers (4)

Train AI models for **specific professional domains**:

| Tokenizer | Vocab | Best For | Downloaded |
|-----------|-------|----------|------------|
| **BioBERT** 🧬 | 28,996 | Biomedical, healthcare, medical research | ✅ Yes |
| **SciBERT** 🔬 | 31,090 | Scientific papers, academic research | ✅ Yes |
| **Legal-BERT** ⚖️ | 30,522 | Legal documents, contracts, compliance | ⚠️ No |
| **FinBERT** 💰 | 30,873 | Financial analysis, trading, markets | ⚠️ No |

---

## 📊 Complete Tokenizer Collection

**Total: 16 Tokenizers** across all categories!

### General Purpose (4)
- ⭐ Qwen 2.5 - 151K vocab (Best overall, 2025)
- Llama 3 - 128K vocab (Requires auth)
- Mistral 7B - 32K vocab
- GPT-2 - 50K vocab (Legacy)

### Code Specialized (3)
- ⭐ DeepSeek-Coder V2 - 100K vocab (Best for code, 2025)
- StarCoder2 - 49K vocab (600+ languages)
- Code Llama - 32K vocab (Stable)

### Vision & Multimodal (3) 🆕
- **CLIP (Vision-Language)** - 49K vocab (Image-text)
- **LLaVA 1.5** - 32K vocab (Visual chat)
- **SigLIP** - 32K vocab (Advanced vision)

### Specialized Domains (4) 🆕
- **BioBERT** - 29K vocab (Biomedical)
- **SciBERT** - 31K vocab (Scientific)
- **Legal-BERT** - 31K vocab (Legal)
- **FinBERT** - 31K vocab (Financial)

### Compact & Efficient (2)
- Phi-3 Mini - 32K vocab
- GPT-2 Base Model - 50K vocab (Backward compatible)

---

## 🎨 Vision & Multimodal Use Cases

### CLIP (Vision-Language)

**What it does:**
- Understands relationships between images and text
- Zero-shot image classification
- Image-text retrieval
- Image generation guidance

**Training Examples:**
- Image captioning datasets
- Visual question answering
- Image-text matching
- Multimodal embeddings

**Best for:**
- Building image search engines
- Visual understanding systems
- Text-to-image generation
- Multimodal AI assistants

**Model Architecture:**
```
Input: Image + Text
      ↓
CLIP Tokenizer (49,408 vocab)
      ↓
Dual Encoders (Vision + Language)
      ↓
Shared Embedding Space
      ↓
Output: Similarity Scores / Embeddings
```

### LLaVA 1.5 (Vision Chat)

**What it does:**
- Visual question answering
- Image understanding and description
- Multimodal conversation
- Visual reasoning

**Training Examples:**
- VQA datasets (Visual Question Answering)
- Image description pairs
- Visual instruction following
- Multimodal dialogue

**Best for:**
- Visual chatbots
- Image analysis assistants
- Accessibility tools (image description)
- Educational AI tutors

**Model Architecture:**
```
Input: Image + Question
      ↓
Vision Encoder (ViT) + LLaVA Tokenizer (32K)
      ↓
Multimodal Fusion
      ↓
Language Model Decoder
      ↓
Output: Text Answer
```

### SigLIP (Sigmoid Loss Vision)

**What it does:**
- Improved vision-language alignment
- Better zero-shot classification
- More efficient training (sigmoid loss vs softmax)
- Enhanced image understanding

**Training Examples:**
- Large-scale image-text pairs
- Visual classification tasks
- Image retrieval datasets
- Cross-modal learning

**Best for:**
- Advanced vision-language models
- Efficient multimodal training
- Zero-shot vision tasks
- Production vision systems

---

## 🔬 Specialized Domain Use Cases

### BioBERT (Biomedical) 🧬

**What it does:**
- Understanding medical terminology
- Biomedical named entity recognition
- Clinical text analysis
- Drug-disease relationship extraction

**Training Examples:**
- PubMed abstracts
- Clinical notes
- Medical research papers
- Drug interaction databases
- Disease symptom datasets

**Best for:**
- Medical AI assistants
- Clinical decision support
- Biomedical research tools
- Healthcare chatbots
- Drug discovery AI

**Pre-trained on:**
- 4.5B words from PubMed abstracts
- 13.5B words from PMC full-text articles

### SciBERT (Scientific) 🔬

**What it does:**
- Understanding scientific terminology
- Research paper analysis
- Citation relationship extraction
- Scientific entity recognition

**Training Examples:**
- Academic papers (1.14M papers)
- Scientific abstracts
- Research datasets
- Technical documentation
- Lab reports

**Best for:**
- Research paper summarization
- Literature review automation
- Scientific question answering
- Academic writing assistance
- Grant proposal analysis

**Pre-trained on:**
- 1.14M papers from Semantic Scholar
- 18% computer science papers
- 82% biomedical papers

### Legal-BERT (Legal) ⚖️

**What it does:**
- Understanding legal terminology
- Contract analysis
- Legal document classification
- Case law reasoning

**Training Examples:**
- Legal contracts
- Court opinions
- Legislation text
- Legal briefs
- Regulatory documents

**Best for:**
- Contract review automation
- Legal research assistants
- Compliance checking
- Document classification
- Legal chatbots

**Pre-trained on:**
- Legal corpora
- Case law databases
- Legislation texts

### FinBERT (Financial) 💰

**What it does:**
- Financial sentiment analysis
- Market trend prediction
- Financial entity recognition
- Risk assessment

**Training Examples:**
- Financial news articles
- Earnings reports
- Market analysis
- Trading data
- Economic indicators

**Best for:**
- Trading algorithms
- Financial news analysis
- Risk assessment tools
- Investment research
- Financial chatbots

**Pre-trained on:**
- Financial news and reports
- Corporate filings
- Market commentary

---

## 🎯 Choosing the Right Tokenizer

### For Image Understanding & Generation
**Use: CLIP** 🖼️
- Large vocabulary (49K)
- Industry-standard vision-language model
- Zero-shot capabilities
- Compatible with Stable Diffusion and other image models

**Example Projects:**
- Image search engines
- Content moderation
- Visual similarity matching
- Text-to-image applications

### For Visual Question Answering
**Use: LLaVA 1.5** 💬
- Optimized for visual chat
- Instruction following
- Multimodal conversation
- Image understanding

**Example Projects:**
- Visual assistants
- Image description tools
- Educational tutors
- Accessibility applications

### For Biomedical AI
**Use: BioBERT** 🧬
- Medical terminology expertise
- Clinical text understanding
- Biomedical entity recognition

**Example Projects:**
- Clinical decision support
- Medical record analysis
- Drug discovery
- Healthcare chatbots

### For Scientific Research
**Use: SciBERT** 🔬
- Scientific vocabulary
- Research paper understanding
- Technical documentation

**Example Projects:**
- Literature review tools
- Research assistants
- Paper summarization
- Citation analysis

### For Legal Applications
**Use: Legal-BERT** ⚖️
- Legal terminology
- Contract understanding
- Case law analysis

**Example Projects:**
- Contract review
- Legal research tools
- Compliance checking
- Document classification

### For Financial Analysis
**Use: FinBERT** 💰
- Financial terminology
- Market sentiment
- Economic understanding

**Example Projects:**
- Trading algorithms
- Market analysis
- Risk assessment
- Financial news processing

---

## 🚀 Quick Start Examples

### Creating a Vision AI Brain

```bash
# 1. Download CLIP tokenizer
python scripts/download_tokenizer.py clip-vit

# 2. Open AI-OS GUI
# 3. HRM Training → Create New

# 4. Configure:
Preset: 10M (or preferred size)
Name: vision-understanding-v1
Goal: Understand and describe images, answer visual questions
Tokenizer: ✓ CLIP (Vision-Language) (49,408 tokens)

# 5. Create & Train on image-text datasets!
```

### Creating a Medical AI Brain

```bash
# 1. Download BioBERT tokenizer
python scripts/download_tokenizer.py biobert

# 2. Configure brain:
Name: medical-assistant-v1
Goal: Analyze medical texts and provide clinical insights
Tokenizer: ✓ BioBERT (Biomedical) (28,996 tokens)

# 3. Train on medical datasets (PubMed, clinical notes, etc.)
```

### Creating a Scientific Research Brain

```bash
# 1. Download SciBERT tokenizer
python scripts/download_tokenizer.py scibert

# 2. Configure brain:
Name: research-assistant-v1
Goal: Understand scientific papers and answer research questions
Tokenizer: ✓ SciBERT (Scientific) (31,090 tokens)

# 3. Train on scientific papers and technical docs
```

---

## 📚 Training Dataset Recommendations

### For Vision Models (CLIP, LLaVA, SigLIP)

**Public Datasets:**
- **COCO** - Image captioning (330K images)
- **Flickr30k** - Image descriptions (31K images)
- **Visual Genome** - Dense annotations (108K images)
- **Conceptual Captions** - 3.3M image-text pairs
- **VQA v2** - Visual question answering (1M+ questions)

**Custom Data:**
- Product images + descriptions
- Screenshots + instructions
- Medical images + diagnoses
- Satellite imagery + labels

### For Biomedical (BioBERT)

**Public Datasets:**
- **PubMed Abstracts** - Medical research
- **MIMIC-III** - Clinical notes (requires approval)
- **PMC-OA** - Full-text biomedical articles
- **ChEMBL** - Drug-target interactions
- **UMLS** - Medical terminology

### For Scientific (SciBERT)

**Public Datasets:**
- **ArXiv** - Scientific papers
- **Semantic Scholar** - Academic papers
- **PubMed Central** - Biomedical literature
- **CiteSeer** - Computer science papers
- **IEEE Xplore** - Engineering papers

### For Legal (Legal-BERT)

**Public Datasets:**
- **EDGAR** - SEC filings
- **Court Listener** - Court opinions
- **EUR-Lex** - EU legal documents
- **CaseLaw Access Project** - US case law

### For Financial (FinBERT)

**Public Datasets:**
- **Financial PhraseBank** - Sentiment analysis
- **StockTwits** - Market sentiment
- **SEC Filings** - Corporate documents
- **Reuters Financial News**
- **Bloomberg News Archives**

---

## 🔄 Multimodal Training Pipeline

### Vision-Language Training Flow

```
1. Prepare Dataset
   ├── Images (.jpg, .png)
   ├── Captions/Labels (.txt, .json)
   └── Pair them correctly

2. Choose Tokenizer
   └── CLIP, LLaVA, or SigLIP

3. Create Brain
   └── Select vision tokenizer in GUI

4. Configure Training
   ├── Image preprocessing (resize, normalize)
   ├── Text tokenization
   └── Batch pairing (image + text)

5. Train Model
   ├── Vision encoder learns image features
   ├── Language encoder learns text features
   └── Contrastive loss aligns both spaces

6. Evaluate
   ├── Image retrieval accuracy
   ├── Text retrieval accuracy
   └── Zero-shot classification
```

---

## 🧪 Testing Results

### All New Tokenizers Verified ✅

```
📊 Vision & Multimodal
─────────────────────────────
✅ CLIP: Downloaded & Tested
✅ LLaVA 1.5: Downloaded & Tested
✅ SigLIP: Downloaded (verification warning)

📊 Specialized Domains
─────────────────────────────
✅ BioBERT: Downloaded & Tested
✅ SciBERT: Downloaded & Tested
⚠️ Legal-BERT: Not yet downloaded
⚠️ FinBERT: Not yet downloaded
```

### Test Brains Created

| Brain Name | Tokenizer | Use Case |
|------------|-----------|----------|
| CLIP-(Vision-Language)-test | clip-vit | Image-text understanding |
| LLaVA-1.5-(Vision-Chat)-test | llava-1.5 | Visual Q&A |
| BioBERT-(Biomedical)-test | biobert | Medical text |
| SciBERT-(Scientific)-test | scibert | Research papers |

---

## 💡 Advanced Use Cases

### Robotics Training (Future)

While we don't have RT-2 tokenizer yet, you can prepare for robotics:

**Using Current Tokenizers:**
1. **Code tokenizers** (DeepSeek-Coder V2, StarCoder2) for action sequences
2. **Vision tokenizers** (CLIP, LLaVA) for visual perception
3. **Qwen 2.5** for multimodal reasoning

**Action Tokenization Concept:**
```python
# Represent robot actions as tokens
robot_actions = {
    "move_forward": 1001,
    "move_backward": 1002,
    "turn_left": 1003,
    "turn_right": 1004,
    "grasp": 1005,
    "release": 1006,
    # ... etc
}

# Train on: Vision → Action Sequences
Input: [image_tokens] + [instruction_tokens]
Output: [action_token_1, action_token_2, ..., action_token_n]
```

### Video Understanding

**Using Vision Tokenizers:**
- CLIP or LLaVA can process video frames
- Tokenize each frame individually
- Temporal modeling at model level

**Workflow:**
```
Video → Extract Frames (e.g., 1 fps)
      ↓
Each Frame → Vision Tokenizer
      ↓
Sequence of Frame Tokens
      ↓
Temporal Model (Transformer)
      ↓
Video Understanding
```

### Audio-Visual Models

**Multimodal Combination:**
```
Audio: Use text tokenizer for transcribed speech
Vision: Use CLIP/LLaVA for visual frames
Fusion: Combine at embedding level

Input: [audio_transcript_tokens] + [video_frame_tokens]
Output: Multimodal understanding
```

---

## 📦 Installation Summary

### Downloaded & Ready (13 Tokenizers)
```bash
✅ gpt2, gpt2-base-model
✅ qwen2.5-7b (Best overall)
✅ mistral-7b
✅ deepseek-coder-v2 (Best for code)
✅ starcoder2
✅ codellama
✅ phi3-mini
✅ clip-vit (Vision-language)
✅ llava-1.5 (Vision chat)
✅ siglip (Advanced vision)
✅ biobert (Biomedical)
✅ scibert (Scientific)
```

### Available to Download (3 Tokenizers)
```bash
⚠️ llama3-8b (requires HF access approval)
⚠️ legal-bert
⚠️ finbert
```

---

## 🎓 Learning Resources

### Vision-Language Models
- **CLIP Paper:** https://arxiv.org/abs/2103.00020
- **LLaVA Paper:** https://arxiv.org/abs/2304.08485
- **SigLIP:** https://arxiv.org/abs/2303.15343

### Domain-Specific Models
- **BioBERT:** https://arxiv.org/abs/1901.08746
- **SciBERT:** https://arxiv.org/abs/1903.10676
- **Legal-BERT:** https://arxiv.org/abs/2010.02559
- **FinBERT:** https://arxiv.org/abs/1908.10063

### Robotics & Embodied AI
- **RT-2:** https://robotics-transformer2.github.io/
- **PaLM-E:** https://palm-e.github.io/

---

## ❓ FAQ

**Q: Can I train image generation models with CLIP?**  
A: CLIP is for understanding, but can guide generation (like in Stable Diffusion). For generation, you'd need image decoder models.

**Q: Do I need special hardware for vision models?**  
A: Vision models need more VRAM. Recommend 8GB+ GPU for small models, 24GB+ for larger ones.

**Q: Can I mix tokenizers (e.g., vision + text)?**  
A: Not directly. Choose one tokenizer per brain. For multimodal, use tokenizers designed for both (CLIP, LLaVA).

**Q: Are specialized tokenizers better than general ones?**  
A: For their specific domains, YES! BioBERT understands medical terms much better than GPT-2.

**Q: Can I train on video datasets?**  
A: Yes! Extract frames, tokenize each frame, and model temporal relationships at the architecture level.

**Q: How do I prepare medical/scientific datasets?**  
A: Start with public datasets (PubMed, ArXiv), then add your own domain-specific data.

**Q: Will you add RT-2 for robotics?**  
A: RT-2 requires special action tokenization. We're exploring options for embodied AI support!

---

## 🎉 Summary

You now have access to **16 specialized tokenizers** covering:

- ✅ **General AI** (Qwen 2.5, Llama 3, Mistral)
- ✅ **Code** (DeepSeek-Coder V2, StarCoder2, CodeLlama)
- ✅ **Vision & Multimodal** (CLIP, LLaVA, SigLIP) 🆕
- ✅ **Biomedical** (BioBERT) 🆕
- ✅ **Scientific** (SciBERT) 🆕
- ✅ **Legal** (Legal-BERT - available) 🆕
- ✅ **Financial** (FinBERT - available) 🆕

**Ready to train specialized AI for any domain!** 🚀

---

*Last Updated: October 13, 2025*  
*Vision and specialized tokenizers fully integrated*

---

## 🧭 Planned Expansion: Image & Video Generation (Design Spec)

> Status: PLANNED – Architecture and UX defined below. No runtime support in main yet.

This addendum extends the tokenizer roadmap to full multimodal generation and scientific-output workflows. It defines the architecture, data contracts, and UI spec required to support:

- Text-to-image, image-to-image, inpainting/outpainting, ControlNet-guided image generation
- Image-to-video and text-to-video (short clips) generation
- Scientific outputs in chat: rich plots, tables, LaTeX, HTML reports, downloadable files
- A new Chat Window v2 that can render most multimodal outputs

### High-level Phases

1) Foundations: artifact store, rich message schema, viewers (images/tables/plots/latex)  
2) Image generation (inference-first): SDXL/LCM, ControlNet, prompt/seed reproducibility  
3) Video generation (inference-first): SVD (image→video), lightweight text→video integration  
4) Scientific outputs: plot/table/LaTeX/HTML/file attachments, export flows  
5) Optional training: dreambooth/LoRA for images; fine-tuning where feasible for video

---

## 🖼️ Image Generation (Planned)

Supported (Planned) pipelines and modalities:

- Text→Image: SD 1.5 / SDXL via Diffusers; optional LCM for fast steps
- Image→Image: style transfer/variations with strength control
- Inpainting/Outpainting: masked editing
- Conditioning: ControlNet (canny, depth, pose), IP-Adapter (face/style), LoRA adapters

Implementation notes:
- Runtime: Hugging Face Diffusers (+ transformers, accelerate, safetensors, xformers)
- Precision: fp16/bf16 on GPU; CPU fallback (slow) for dev
- Reproducibility: seed, guidance_scale, scheduler stored with artifact metadata
- Safety filter hooks: optional NSFW classifier toggle (user-controlled)

CLI (planned examples):
```
aios gen image --model sdxl --prompt "a red fox in a misty forest, photorealistic" \
      --steps 20 --seed 42 --size 1024x1024 --guidance-scale 7.5 \
      --out artifacts/outputs/images/

aios gen image2image --model sd15 --init-image path/to/img.png --strength 0.6 \
      --prompt "studio portrait, soft lighting" --lora face_finetune.safetensors
```

---

## 🎞️ Video Generation (Planned)

Initial scope focuses on inference and short clips:

- Image→Video: Stable Video Diffusion (SVD) 14-24 fps, 2-4s clips
- Text→Image→Video: generate keyframe with SDXL then animate via SVD
- Text→Video (experimental): integrate an open model (e.g., CogVideoX small) when stable

Key controls:
- fps (e.g., 8–24), duration (seconds), resolution (e.g., 576p/720p), motion strength
- Seed and scheduler captured for reproducibility

CLI (planned examples):
```
aios gen video --from-image artifacts/outputs/images/fox.jpg --model svd --fps 14 --seconds 3

aios gen video --prompt "drone shot over snowy mountains at sunrise" \
      --pipeline txt2img+svd --fps 12 --seconds 4 --seed 123
```

Dataset prep (future training): frame extraction to frames/, JSON pairs with timing; ffmpeg helpers.

---

## 🧪 Scientific Output Interfaces (Planned)

The assistant will produce structured scientific artifacts alongside text:

- Plots: Plotly JSON spec (preferred for interactivity) or Matplotlib PNG fallback
- Tables: column schema + row data; optional CSV/Parquet attachment for download
- Equations: LaTeX rendered with KaTeX in UI
- Reports: minimal HTML (sanitized) and Markdown export
- Files: CSV/JSON/NetCDF/HDF5/ZIP as downloadable artifacts
- 3D/Geo (future): GLB/PLY previews; GeoJSON/tiles via map component (opt-in)

Server responsibilities:
- Validate payload sizes and types; store artifacts with content hash
- Generate preview thumbnails/posters where applicable
- Attach metadata (units, provenance, seeds, code refs)

---

## 🧩 Chat Window v2 – Rich Message Schema (Planned)

The chat UI will display multi-part messages. Each message may contain text plus a list of parts. Parts reference inline payloads or stored artifacts.

Message (concept):
```json
{
      "id": "msg_01",
      "role": "assistant",
      "parts": [
            { "type": "text", "text": "Here is the generated image." },
            { "type": "image", "source": { "type": "artifact", "id": "art_abc" }, "alt": "red fox" },
            { "type": "plot", "spec": { "version": 2, "data": [], "layout": {} } },
            { "type": "table", "columns": [{"name":"time","type":"number"}], "rows": [[0],[1]],
                  "download": { "artifactId": "art_tbl" } },
            { "type": "latex", "code": "E=mc^2" },
            { "type": "file", "artifactId": "art_zip", "name": "results.zip" }
      ]
}
```

Artifact (concept):
```json
{
      "id": "art_abc",
      "kind": "image",
      "mime": "image/png",
      "path": "artifacts/outputs/images/fox.png",
      "sha256": "...",
      "width": 1024,
      "height": 1024,
      "createdAt": "2025-10-15T12:34:56Z",
      "metadata": {
            "prompt": "a red fox in a misty forest",
            "seed": 42,
            "steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "EulerA"
      }
}
```

UI Behaviors:
- Text streams first; media parts render when artifacts are ready
- Image/video gallery with zoom; video player with poster/thumb
- Plotly interactive viewer; CSV download from table; LaTeX inline via KaTeX
- Side “Artifacts” panel lists recent outputs with search and filters
- Tool-run logs collapsible; copy-to-clipboard for prompts/configs

Security & Limits:
- Sanitize HTML; disallow inline scripts; CSP headers
- Size limits per part and per message; chunked uploads for large files

---

## 🏗️ Architecture Addendum (Planned)

Logical components:

1) Generation Runtimes
       - Diffusers pipelines: SD 1.5, SDXL, SVD (+ ControlNet, IP-Adapter, LoRA)
       - Scheduler registry; VRAM-aware autotune
2) Artifact Store
       - Path: `artifacts/outputs/{images|videos|plots|tables|files}/` (content-hashed)
       - Metadata JSON sidecars; thumbnail/poster derivation jobs
3) Messaging API
       - Returns messages with parts; emits events: text-delta, artifact-pending, artifact-ready
4) Chat UI v2
       - React components: Text, Image, Video, Plotly, Table, LaTeX, File
       - View toggles: compact/threaded/gallery
5) CLI/HRM Integrations
       - `aios gen image|video` commands; HRM tool to attach outputs to runs

Deployment considerations:
- Windows, Linux GPU; optional CPU fallback for dev  
- CUDA/cuDNN versions pinned; xformers optional  
- Model weights cached in `training_data/hf_cache/`

---

## 📐 Data Contracts (Planned)

MessagePart types:
- text, image, video, audio (future), plot, table, latex, html (sanitized), file

Common fields:
- `type`, `source` (inline|artifact), `mime` (when applicable), `metadata`

Table contract (sketch):
```json
{
      "type": "table",
      "columns": [ {"name":"col","type":"string","unit":""} ],
      "rows": [ ["value"] ],
      "download": { "artifactId": "art_csv" }
}
```

Plotly contract: store full `spec` and optional `screenshot` thumbnail.

---

## 🧪 Acceptance Criteria (Milestones)

M1 – Foundations
- Render images, tables (CSV download), Plotly, LaTeX in Chat v2
- Artifact store with hashing + metadata; events wired

M2 – Image Generation (Inference)
- SDXL text→image end-to-end via CLI and Chat tool
- Reproducible artifacts (seed/scheduler stored); ControlNet optional

M3 – Video Generation (Inference)
- SVD image→video; keyframe path (txt→img→vid) exposed in CLI and Chat
- Video player in UI with fps/duration metadata

M4 – Scientific Outputs
- Plot/table/latex produced by tools; downloadable CSV/ZIP
- Simple HTML report (sanitized) preview and export

M5 – UX Polish & Reliability
- Gallery view, artifact panel, retries, error toasts, rate limits

---

## 🛠️ Dependencies (Planned)

Python
- diffusers, transformers, accelerate, torch, safetensors, xformers (optional)
- pillow, opencv-python, ffmpeg-python (or system ffmpeg)

Frontend
- KaTeX, Plotly.js, React Player (or video element), MIME renderers

---

## ⚠️ Risks & Considerations

- VRAM constraints: add auto-downscale and low-VRAM schedulers
- Large artifacts: background upload and streaming; disk quotas
- Content safety: optional NSFW filters; user control and transparency
- Licensing: check weights and dataset licenses; document restrictions

---

## 🧪 Try-it (Future, subject to change)

```
# Text → Image
aios gen image --model sdxl --prompt "scientific diagram of DNA helix, vector style" --size 1024x1024

# Image → Video
aios gen video --from-image artifacts/outputs/images/diagram.png --model svd --fps 12 --seconds 3

# Scientific Plot in Chat (tool)
aios tools plot --spec path/to/plotly.json --attach
```

---

This specification updates the planned feature set to include multimodal generation and scientific outputs with a concrete data and UI contract, without claiming availability today.
