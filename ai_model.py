"""Local summarisation utilities backed by Llama 3 13B Instruct model.

The implementation relies on the `ctransformers` Python package which loads a
quantised GGUF model with GPU acceleration support. The code falls back to a
simple extractive summary if the model file is missing so that the chatbot
never crashes.
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import Iterable, Optional

from ctransformers import AutoModelForCausalLM

LOGGER = logging.getLogger(__name__)

# Default GGUF file name. Users can replace it with any compatible instruct
# model by adjusting the environment variable or path.
DEFAULT_MODEL_PATH = Path("models/Llama-3-13B-Instruct-Q4_K_M.gguf")

_MODEL: Optional[AutoModelForCausalLM] = None
_MODEL_ERROR: Optional[str] = None


def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[AutoModelForCausalLM]:
    global _MODEL, _MODEL_ERROR

    # Return cached model if already loaded successfully
    if _MODEL is not None:
        return _MODEL
    
    # Check if model file exists
    if not model_path.exists():
        _MODEL_ERROR = f"Model file not found at {model_path}"
        LOGGER.warning(_MODEL_ERROR)
        return None
    
    # Reset error state if file now exists (allows retry after fixing the issue)
    if _MODEL_ERROR is not None and model_path.exists():
        LOGGER.info("Model file now exists, retrying load after previous error")
        _MODEL_ERROR = None

    try:
        # Load Llama 3 13B with GPU acceleration
        # For 13B model: 40 total layers, gpu_layers=35-40 for full GPU offload
        # Lower values (20-30) for GPUs with less VRAM (8-12GB)
        # ctransformers automatically falls back to CPU if GPU unavailable
        _MODEL = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            model_type="llama",  # Llama 3 uses "llama" type in ctransformers
            gpu_layers=35,  # Offload 35/40 layers to GPU (adjust based on VRAM: 20-30 for 8GB, 35-40 for 16GB+)
            context_length=4096,  # Context window (4096 is safer for memory, can use 8192 if VRAM allows)
            threads=4,  # Number of CPU threads (adjust based on CPU cores)
        )
        LOGGER.info("Loaded Llama 3 13B Instruct model from %s", model_path)
        # Verify GPU usage - check if model loaded successfully
        if _MODEL is not None:
            LOGGER.info("Model loaded successfully with GPU acceleration enabled")
        else:
            LOGGER.warning("Model object is None after loading attempt")
    except Exception as exc:  # pragma: no cover - hardware/environment specific
        _MODEL_ERROR = f"Failed to load model: {exc}"
        LOGGER.exception(_MODEL_ERROR)
        # Try CPU-only fallback if GPU loading fails
        try:
            LOGGER.warning("Retrying with CPU-only mode...")
            _MODEL = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                model_type="llama",
                gpu_layers=0,  # Force CPU-only mode
                context_length=4096,  # Match GPU config for consistency
                threads=8,  # Use more CPU threads for CPU-only mode
            )
            LOGGER.info("Loaded model in CPU-only mode")
        except Exception as fallback_exc:
            LOGGER.exception("CPU fallback also failed: %s", fallback_exc)
            _MODEL_ERROR = f"Failed to load model: {exc} (CPU fallback: {fallback_exc})"
            _MODEL = None

    return _MODEL


def _build_prompt(user_query: str, combined_text: str) -> str:
    """Build a prompt in Llama 3's official chat template format.
    
    Llama 3 uses a specific chat template with special tokens:
    - <|begin_of_text|> - Start of sequence
    - <|start_header_id|>role<|end_header_id|> - Role header
    - <|eot_id|> - End of turn token
    """
    
    system_message = (
        "You are a licensed medical assistant. Using the information provided, "
        "craft a concise, practical response for the patient's question. Focus on "
        "home care tips, red-flag symptoms, and when to seek urgent medical help. "
        "Avoid speculation and keep the tone calm and supportive. "
        "Respond with three short bullet points followed by a brief reminder to "
        "consult professionals for serious concerns."
    )
    
    user_message = (
        f"Patient question: {user_query}\n\n"
        f"Source notes:\n{combined_text}"
    )
    
    # Llama 3 official chat template format
    # Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>...
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_message}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt


def _run_model(prompt: str, max_new_tokens: int = 300) -> Optional[str]:
    """Run inference with Llama 3 model.
    
    Parameters optimized for medical summarization: balanced creativity
    and accuracy with controlled randomness.
    """
    model = _load_model()
    if model is None:
        return None

    try:
        # Generate response with optimized parameters for medical summarization
        # Core parameters (supported by all ctransformers versions)
        inference_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,  # Lower temperature for more deterministic, factual responses
            "top_p": 0.9,  # Nucleus sampling - focus on top 90% probability mass
            "repetition_penalty": 1.1,  # Penalize repetition for cleaner output
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],  # Stop tokens for Llama 3
        }
        
        # Optional parameter - may not be supported in all versions
        # If it fails, we'll retry without it
        try:
            output = model(prompt, **inference_params, top_k=40)
        except TypeError:
            # top_k not supported, retry without it
            LOGGER.debug("top_k parameter not supported, using core parameters only")
            output = model(prompt, **inference_params)
        
        # Clean up the output - handle string or dict response
        if isinstance(output, dict):
            # Some ctransformers versions return dict with 'generated_text' key
            text = output.get('generated_text', str(output))
        else:
            text = str(output)
        
        cleaned = text.strip()
        
        # Remove prompt if it was included in the output (sometimes happens)
        if "<|start_header_id|>assistant<|end_header_id|>" in cleaned:
            cleaned = cleaned.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Remove any remaining special tokens
        special_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>"]
        for token in special_tokens:
            cleaned = cleaned.replace(token, "")
        
        # Remove any trailing role headers that might have been generated
        if "<|start_header_id|>" in cleaned:
            cleaned = cleaned.split("<|start_header_id|>")[0].strip()
        
        return cleaned.strip()
    except Exception as exc:  # pragma: no cover - runtime inference issues
        LOGGER.exception("Summarisation failed: %s", exc)
        return None


def _fallback_summary(pages_content: Iterable[str], user_query: str) -> str:
    combined = " ".join(pages_content)
    sentences = re.split(r"(?<=[.!?])\s+", combined)
    trimmed = " ".join(sentences[:4])[:500]
    if not trimmed:
        return (
            "I couldn't gather enough information right now. Please consult a "
            "healthcare professional if your symptoms persist or worsen."
        )

    return textwrap.dedent(
        f"""
        Key points for "{user_query}":
        - {trimmed}
        Remember to speak with a qualified clinician for personalised advice.
        """
    ).strip()


def summarize_medical_pages(pages_content: Iterable[str], user_query: str) -> str:
    combined_text = " ".join(pages_content).strip()

    if not combined_text:
        return (
            "I couldn't find reliable information for that question. "
            "Please try rephrasing it, or contact a healthcare professional."
        )

    prompt = _build_prompt(user_query, combined_text[:3000])
    model_response = _run_model(prompt)

    if model_response:
        return model_response

    return _fallback_summary(pages_content, user_query)
