"""Central registry for available tokenizers in AI-OS.

This module provides a unified interface for managing multiple tokenizers,
allowing users to select the optimal tokenizer for their use case during
brain creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TokenizerInfo:
    """Metadata for a tokenizer."""
    
    id: str
    """Unique identifier (e.g., 'llama3-8b')"""
    
    name: str
    """Display name for UI"""
    
    vocab_size: int
    """Vocabulary size (number of tokens)"""
    
    path: str
    """Local path or HuggingFace model ID"""
    
    description: str
    """User-friendly description"""
    
    compression_ratio: float
    """Average characters per token (efficiency metric)"""
    
    recommended_for: list[str]
    """Use cases this tokenizer excels at"""
    
    requires_auth: bool
    """Whether downloading requires HuggingFace authentication"""
    
    hf_model_id: Optional[str] = None
    """HuggingFace model ID for downloading (if different from path)"""


class TokenizerRegistry:
    """Central registry of available tokenizers.
    
    This class maintains a catalog of tokenizers that can be used for
    brain creation. Each tokenizer has different characteristics and
    is optimized for different use cases.
    
    Example:
        >>> registry = TokenizerRegistry()
        >>> tokenizers = registry.list_available()
        >>> default = registry.get_default()
        >>> info = registry.get("llama3-8b")
        >>> installed = registry.check_installed("llama3-8b")
    """
    
    TOKENIZERS = {
        # === Legacy Tokenizers ===
        "gpt2": TokenizerInfo(
            id="gpt2",
            name="GPT-2 (Legacy)",
            vocab_size=50257,
            path="artifacts/hf_implant/tokenizers/gpt2",
            description="Original GPT-2 tokenizer. Good for English text, proven and reliable.",
            compression_ratio=3.5,
            recommended_for=["english-text", "legacy", "testing", "small-models"],
            requires_auth=False,
            hf_model_id="gpt2",
        ),
        "gpt2-base-model": TokenizerInfo(
            id="gpt2-base-model",
            name="GPT-2 (Base Model - Current Default)",
            vocab_size=50257,
            path="artifacts/hf_implant/base_model",
            description="Current default GPT-2 tokenizer. Same as 'gpt2' but using existing base_model path.",
            compression_ratio=3.5,
            recommended_for=["english-text", "legacy", "backward-compatibility"],
            requires_auth=False,
            hf_model_id="gpt2",
        ),
        
        # === Latest General Purpose (2025) ===
        "qwen2.5-7b": TokenizerInfo(
            id="qwen2.5-7b",
            name="Qwen 2.5 (Latest, Best Overall)",
            vocab_size=151643,
            path="artifacts/hf_implant/tokenizers/qwen2.5-7b",
            description="Latest 2025 tokenizer with massive vocabulary. Exceptional multilingual, code, and math capabilities. Best overall choice.",
            compression_ratio=5.2,
            recommended_for=["chat", "multilingual", "code", "math", "reasoning", "general"],
            requires_auth=False,
            hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        ),
        "mistral-7b": TokenizerInfo(
            id="mistral-7b",
            name="Mistral 7B",
            vocab_size=32000,
            path="artifacts/hf_implant/tokenizers/mistral-7b",
            description="Balanced efficiency with good compression. Excellent for chat and general tasks.",
            compression_ratio=4.2,
            recommended_for=["chat", "efficiency", "general", "balanced"],
            requires_auth=False,
            hf_model_id="mistralai/Mistral-7B-v0.1",
        ),
        
        # === Code-Specialized Tokenizers ===
        "deepseek-coder-v2": TokenizerInfo(
            id="deepseek-coder-v2",
            name="DeepSeek-Coder V2 (Best for Code)",
            vocab_size=100000,
            path="artifacts/hf_implant/tokenizers/deepseek-coder-v2",
            description="State-of-the-art 2025 code tokenizer with enhanced math reasoning. Best for programming tasks.",
            compression_ratio=5.0,
            recommended_for=["code", "programming", "math", "reasoning", "algorithms"],
            requires_auth=False,
            hf_model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        ),
        "starcoder2": TokenizerInfo(
            id="starcoder2",
            name="StarCoder2 (Code Generation)",
            vocab_size=49152,
            path="artifacts/hf_implant/tokenizers/starcoder2",
            description="Latest BigCode tokenizer optimized for code generation across 600+ programming languages.",
            compression_ratio=4.7,
            recommended_for=["code", "programming", "multi-language-code", "fill-in-middle"],
            requires_auth=False,
            hf_model_id="bigcode/starcoder2-15b",
        ),
        "codellama": TokenizerInfo(
            id="codellama",
            name="Code Llama (Stable)",
            vocab_size=32016,
            path="artifacts/hf_implant/tokenizers/codellama",
            description="Meta's code tokenizer. Reliable and well-tested for programming tasks.",
            compression_ratio=4.5,
            recommended_for=["code", "programming", "software-development"],
            requires_auth=False,
            hf_model_id="codellama/CodeLlama-7b-hf",
        ),
        
        # === Compact & Efficient ===
        "phi3-mini": TokenizerInfo(
            id="phi3-mini",
            name="Phi-3 Mini",
            vocab_size=32064,
            path="artifacts/hf_implant/tokenizers/phi3-mini",
            description="Microsoft's efficient tokenizer. Good balance of size and capability.",
            compression_ratio=4.0,
            recommended_for=["efficiency", "compact", "general"],
            requires_auth=False,
            hf_model_id="microsoft/Phi-3-mini-4k-instruct",
        ),
        
        # === Vision & Multimodal ===
        "clip-vit": TokenizerInfo(
            id="clip-vit",
            name="CLIP (Vision-Language)",
            vocab_size=49408,
            path="artifacts/hf_implant/tokenizers/clip-vit",
            description="OpenAI's CLIP tokenizer for vision-language tasks. Bridges images and text for multimodal understanding.",
            compression_ratio=3.8,
            recommended_for=["vision-language", "image-text", "multimodal", "image-generation"],
            requires_auth=False,
            hf_model_id="openai/clip-vit-base-patch32",
        ),
        "llava-1.5": TokenizerInfo(
            id="llava-1.5",
            name="LLaVA 1.5 (Vision Chat)",
            vocab_size=32000,
            path="artifacts/hf_implant/tokenizers/llava-1.5",
            description="Vision-language chat model tokenizer. Excellent for image understanding and visual question answering.",
            compression_ratio=4.1,
            recommended_for=["vision-chat", "visual-qa", "image-understanding", "multimodal-chat"],
            requires_auth=False,
            hf_model_id="llava-hf/llava-1.5-7b-hf",
        ),
        "siglip": TokenizerInfo(
            id="siglip",
            name="SigLIP (Sigmoid Loss Vision)",
            vocab_size=32000,
            path="artifacts/hf_implant/tokenizers/siglip",
            description="Google's improved vision-language tokenizer with sigmoid loss. Better than CLIP for many vision tasks.",
            compression_ratio=4.0,
            recommended_for=["vision-language", "image-classification", "zero-shot-vision"],
            requires_auth=False,
            hf_model_id="google/siglip-base-patch16-224",
        ),
        
        # === Specialized Domains ===
        "biobert": TokenizerInfo(
            id="biobert",
            name="BioBERT (Biomedical)",
            vocab_size=28996,
            path="artifacts/hf_implant/tokenizers/biobert",
            description="Specialized for biomedical and healthcare text. Pre-trained on PubMed abstracts and PMC full-text articles.",
            compression_ratio=3.9,
            recommended_for=["biomedical", "healthcare", "scientific", "medical"],
            requires_auth=False,
            hf_model_id="dmis-lab/biobert-v1.1",
        ),
        "legal-bert": TokenizerInfo(
            id="legal-bert",
            name="Legal-BERT (Legal Domain)",
            vocab_size=30522,
            path="artifacts/hf_implant/tokenizers/legal-bert",
            description="Specialized for legal documents and contracts. Trained on legal corpora.",
            compression_ratio=3.7,
            recommended_for=["legal", "contracts", "law", "compliance"],
            requires_auth=False,
            hf_model_id="nlpaueb/legal-bert-base-uncased",
        ),
        "finbert": TokenizerInfo(
            id="finbert",
            name="FinBERT (Financial)",
            vocab_size=30873,
            path="artifacts/hf_implant/tokenizers/finbert",
            description="Specialized for financial text analysis. Pre-trained on financial news and reports.",
            compression_ratio=3.8,
            recommended_for=["finance", "trading", "financial-analysis", "markets"],
            requires_auth=False,
            hf_model_id="ProsusAI/finbert",
        ),
        "scibert": TokenizerInfo(
            id="scibert",
            name="SciBERT (Scientific)",
            vocab_size=31090,
            path="artifacts/hf_implant/tokenizers/scibert",
            description="Specialized for scientific publications. Trained on 1.14M papers from Semantic Scholar.",
            compression_ratio=4.0,
            recommended_for=["scientific", "research", "academic", "papers"],
            requires_auth=False,
            hf_model_id="allenai/scibert_scivocab_uncased",
        ),
    }
    
    @classmethod
    def list_available(cls) -> list[TokenizerInfo]:
        """List all registered tokenizers.
        
        Returns:
            List of TokenizerInfo objects for all available tokenizers.
        """
        return list(cls.TOKENIZERS.values())
    
    @classmethod
    def get(cls, tokenizer_id: str) -> Optional[TokenizerInfo]:
        """Get tokenizer info by ID.
        
        Args:
            tokenizer_id: Unique tokenizer identifier
            
        Returns:
            TokenizerInfo if found, None otherwise
        """
        return cls.TOKENIZERS.get(tokenizer_id)
    
    @classmethod
    def get_default(cls) -> TokenizerInfo:
        """Get recommended default tokenizer.
        
        Returns:
            TokenizerInfo for the recommended default (currently Qwen 2.5)
        """
        return cls.TOKENIZERS["qwen2.5-7b"]
    
    @classmethod
    def get_legacy_default(cls) -> TokenizerInfo:
        """Get legacy default tokenizer (GPT-2 base model).
        
        For backward compatibility with existing brains.
        
        Returns:
            TokenizerInfo for GPT-2 base model
        """
        return cls.TOKENIZERS["gpt2-base-model"]
    
    @classmethod
    def check_installed(cls, tokenizer_id: str, project_root: Optional[str] = None) -> bool:
        """Check if tokenizer files exist locally.
        
        Args:
            tokenizer_id: Unique tokenizer identifier
            project_root: Optional project root directory to resolve relative paths.
                         If not provided, uses current working directory.
            
        Returns:
            True if tokenizer is installed, False otherwise
        """
        info = cls.get(tokenizer_id)
        if not info:
            return False
        
        path = Path(info.path)
        
        # If path is relative and project_root is provided, resolve relative to project_root
        if not path.is_absolute() and project_root:
            path = Path(project_root) / path
        
        if not path.exists():
            return False
        
        # Check for required tokenizer files
        required_files = ["tokenizer_config.json"]
        # At least one of these should exist
        tokenizer_files = ["tokenizer.json", "vocab.json", "tokenizer.model"]
        
        has_required = all((path / f).exists() for f in required_files)
        has_tokenizer = any((path / f).exists() for f in tokenizer_files)
        
        return has_required and has_tokenizer
    
    @classmethod
    def find_by_path(cls, path: str, project_root: Optional[str] = None) -> Optional[TokenizerInfo]:
        """Find tokenizer info by path.
        
        Useful for identifying which tokenizer is used by an existing brain.
        
        Args:
            path: Tokenizer path (exact or partial match)
            project_root: Optional project root directory to resolve relative paths
            
        Returns:
            TokenizerInfo if found, None otherwise
        """
        path_normalized = str(Path(path)).lower().replace("\\", "/")
        
        for info in cls.TOKENIZERS.values():
            info_path = Path(info.path)
            # Resolve relative paths if project_root provided
            if not info_path.is_absolute() and project_root:
                info_path = Path(project_root) / info_path
            info_path_str = str(info_path).lower().replace("\\", "/")
            if info_path_str in path_normalized or path_normalized in info_path_str:
                return info
        
        return None
    
    @classmethod
    def get_installed_tokenizers(cls, project_root: Optional[str] = None) -> list[TokenizerInfo]:
        """Get list of currently installed tokenizers.
        
        Args:
            project_root: Optional project root directory to resolve relative paths
        
        Returns:
            List of TokenizerInfo for installed tokenizers only
        """
        return [info for info in cls.list_available() if cls.check_installed(info.id, project_root)]
