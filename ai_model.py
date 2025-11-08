"""Local summarisation utilities backed by a lightweight Mistral model.

The implementation relies on the `ctransformers` Python package which loads a
quantised GGUF model directly on CPU, avoiding heavyweight dependencies such as
PyTorch. The code falls back to a simple extractive summary if the model file
is missing so that the chatbot never crashes.
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
DEFAULT_MODEL_PATH = Path("models/mistral-7b-instruct.Q4_K_M.gguf")

_MODEL: Optional[AutoModelForCausalLM] = None
_MODEL_ERROR: Optional[str] = None


def _load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Optional[AutoModelForCausalLM]:
    global _MODEL, _MODEL_ERROR

    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL

    if not model_path.exists():
        _MODEL_ERROR = f"Model file not found at {model_path}"
        LOGGER.warning(_MODEL_ERROR)
        return None

    try:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            model_type="mistral",
            gpu_layers=0,
            context_length=4096,
        )
        LOGGER.info("Loaded summarisation model from %s", model_path)
    except Exception as exc:  # pragma: no cover - hardware/environment specific
        _MODEL_ERROR = f"Failed to load model: {exc}"
        LOGGER.exception(_MODEL_ERROR)
        _MODEL = None

    return _MODEL


def _build_prompt(user_query: str, combined_text: str) -> str:
    header = textwrap.dedent(
        f"""
        [INST]
        You are a licensed medical assistant. Using the information provided,
        craft a concise, practical response for the patient's question. Focus on
        home care tips, red-flag symptoms, and when to seek urgent medical help.
        Avoid speculation and keep the tone calm and supportive.

        Patient question: {user_query}
        Source notes:
        {combined_text}

        Respond with three short bullet points followed by a brief reminder to
        consult professionals for serious concerns.
        [/INST]
        """
    ).strip()

    return re.sub(r"\s+", " ", header)


def _run_model(prompt: str, max_new_tokens: int = 220) -> Optional[str]:
    model = _load_model()
    if model is None:
        return None

    try:
        output = model(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        return output.strip()
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
