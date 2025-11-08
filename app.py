"""Streamlit interface for the local medical chatbot."""

from __future__ import annotations

import logging
from typing import List

import streamlit as st

from ai_model import summarize_medical_pages
from scraper import available_sources, scrape_medical_info

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Local Medical Assistant", page_icon="🩺")


def run_app() -> None:
    st.title("🩺 Local Medical Assistant")
    st.write(
        "Get concise, practical advice for common symptoms using trusted WebMD "
        "and Mayo Clinic content. All processing happens locally."
    )

    with st.expander("Trusted sources"):
        for url in available_sources():
            st.markdown(f"- [{url}]({url})")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_area(
        "Describe your symptoms or ask a medical question",
        placeholder="e.g. I'm feeling feverish with a sore throat...",
        height=120,
    )

    if st.button("Get advice"):
        if not user_input.strip():
            st.warning("Please enter a question before requesting advice.")
        else:
            with st.spinner("Analysing trusted sources..."):
                pages_content: List[str] = scrape_medical_info(user_input)

                if not pages_content:
                    st.error(
                        "Unable to retrieve information right now. Try a "
                        "different question or check your internet connection."
                    )
                else:
                    reply = summarize_medical_pages(pages_content, user_input)
                    st.session_state.history.append((user_input, reply))
                    st.success("Advice ready!")

    if st.session_state.history:
        st.divider()
        for question, answer in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_app()
