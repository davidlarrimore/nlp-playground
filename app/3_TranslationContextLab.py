"""Demonstrates contextual translation differences using Streamlit."""

import streamlit as st

from utils import local_translate_to_english, translate_with_source_language, translate_context_to_language


# Pre-defined example sentences in different languages (all with the same ambiguous meaning).
LANGUAGE_EXAMPLES = {
    "Spanish": {
        "text": "Los compañeros protestaron cuando el revisor principal cuestionó la decisión y revirtió el resultado después de consultar con los asesores sobre el enfoque.",
        "language": "Spanish",
        "emoji": "🇪🇸",
        "script": "Latin",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
    "Russian": {
        "text": "Товарищи по команде протестовали, когда главный рецензент оспорил решение и отменил результат после консультации с советниками по поводу подхода.",
        "language": "Russian",
        "emoji": "🇷🇺",
        "script": "Cyrillic",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
    "Arabic": {
        "text": "احتج زملاء الفريق عندما شكك المراجع الرئيسي في القرار وعكس النتيجة بعد التشاور مع المستشارين حول النهج.",
        "language": "Arabic",
        "emoji": "🇸🇦",
        "script": "Arabic (RTL)",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
    "Chinese": {
        "text": "当主审查员质疑决定并在与顾问就方法进行协商后推翻结果时，队友们提出了抗议。",
        "language": "Chinese",
        "emoji": "🇨🇳",
        "script": "Hanzi",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
    "Japanese": {
        "text": "主任審査員が決定に異議を唱え、アドバイザーとアプローチについて相談した後に結果を覆した時、チームメイトたちは抗議しました。",
        "language": "Japanese",
        "emoji": "🇯🇵",
        "script": "Japanese",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
    "Greek": {
        "text": "Οι συμπαίκτες διαμαρτυρήθηκαν όταν ο κύριος ελεγκτής αμφισβήτησε την απόφαση και ανέτρεψε το αποτέλεσμα αφού συμβουλεύτηκε τους συμβούλους σχετικά με την προσέγγιση.",
        "language": "Greek",
        "emoji": "🇬🇷",
        "script": "Greek",
        "english_original": "The teammates protested when the lead reviewer challenged the decision and reversed the outcome after consulting with advisors about the approach.",
        "context_sports": "This is about a professional soccer match where the referee made a controversial offside decision during the game",
        "context_business": "This is about a corporate board meeting where executives reviewed a strategic proposal and rejected the merger recommendation after consulting with financial advisors",
    },
}


def ensure_session_keys() -> None:
    """Initialize translation results storage."""
    keys = [
        "selected_language",
        "selected_text",
        "english_context",
        "translated_context",
        "translation_without_context",
        "translation_with_context",
        "show_results",
        "live_source_text",
        "live_source_lang",
        "live_openai_translation",
        "live_local_translation",
    ]
    for key in keys:
        if key not in st.session_state:
            if key == "show_results":
                st.session_state[key] = False
            else:
                st.session_state[key] = ""


def select_language(lang_key: str) -> None:
    """Handler for language selection."""
    st.session_state["selected_language"] = lang_key
    st.session_state["selected_text"] = LANGUAGE_EXAMPLES[lang_key]["text"]
    # Set default context to sports
    st.session_state["english_context"] = LANGUAGE_EXAMPLES[lang_key]["context_sports"]
    # Reset results
    st.session_state["translated_context"] = ""
    st.session_state["translation_without_context"] = ""
    st.session_state["translation_with_context"] = ""
    st.session_state["show_results"] = False


def translation_page() -> None:
    """Multi-language translation demonstration with context awareness."""
    st.markdown("# 🌐 Translation Context Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Select a language example below to see how context transforms translation quality.
                The sentence contains context-neutral words like "teammates", "reviewer", "decision", and "approach"
                that work in both sports and business contexts. The model will translate your English
                context to the source language, then use it to accurately translate back to English.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This is a neural machine translation (NMT) demo: the model reads the full sentence (and optional context) "
        "at once to produce fluent output. Earlier statistical MT systems relied on phrase tables and often sounded "
        "choppy; use the context toggle below to see how neural models stay fluent while disambiguating meaning."
    )

    ensure_session_keys()

    st.markdown("### ⚡ Live translation (OpenAI vs. local NLP)")
    st.markdown(
        "<p style='color: #6b7280; margin-bottom: 0.5rem;'>Compare your configured OPENAI_TRANSLATION_MODEL with a lightweight, offline baseline (Helsinki-NLP/opus-mt-mul-en). Both translate into English so you can see quality differences.</p>",
        unsafe_allow_html=True,
    )

    col_src, col_lang = st.columns([3, 1])
    with col_src:
        st.session_state["live_source_text"] = st.text_area(
            "Source sentence (any language → English)",
            value=st.session_state.get("live_source_text") or "El clima hoy es hermoso.",
            height=80,
        )
    with col_lang:
        st.session_state["live_source_lang"] = st.selectbox(
            "Source language label (for OpenAI prompt)",
            options=["Spanish", "Russian", "Arabic", "Chinese", "Japanese", "Greek", "Other"],
            index=0,
        )

    if st.button("Translate with both models", use_container_width=True):
        with st.spinner("Translating..."):
            st.session_state["live_openai_translation"] = translate_with_source_language(
                st.session_state["live_source_text"],
                st.session_state["live_source_lang"],
                "English",
            )
            st.session_state["live_local_translation"] = local_translate_to_english(
                st.session_state["live_source_text"]
            )

    if st.session_state.get("live_openai_translation") or st.session_state.get("live_local_translation"):
        col_oai, col_local = st.columns(2)
        with col_oai:
            st.markdown("**OPENAI_TRANSLATION_MODEL**")
            st.success(st.session_state.get("live_openai_translation", ""))
        with col_local:
            st.markdown("**Local baseline (Helsinki-NLP/opus-mt-mul-en)**")
            st.info(st.session_state.get("live_local_translation", ""))

    st.divider()

    # Language Selection Section
    st.markdown("### 🌍 Step 1: Select a Language Example")
    st.markdown(
        "<p style='color: #6b7280; margin-bottom: 1rem;'>Click on a language card to load an example sentence. Each card contains the same sentence about 'teammates', 'reviewer', 'decision', and 'approach' - context-neutral terms that could refer to either a sports match OR a business meeting!</p>",
        unsafe_allow_html=True,
    )

    # Display language cards in a grid
    # Arrange language cards across three responsive columns.
    cols = st.columns(3)
    for idx, (lang_key, lang_data) in enumerate(LANGUAGE_EXAMPLES.items()):
        with cols[idx % 3]:
            # Style the card to highlight whichever language is currently chosen.
            is_selected = st.session_state.get("selected_language") == lang_key

            # Create a clickable button that looks like a card
            button_label = f"{lang_data['emoji']} {lang_data['language']}"

            if st.button(
                button_label,
                key=f"btn_{lang_key}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=f"{lang_data['script']} Script: {lang_data['text']}"
            ):
                select_language(lang_key)
                st.rerun()

            # Display the text sample below the button
            border_color = "#3b82f6" if is_selected else "#e5e7eb"
            background = "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)" if is_selected else "#f9fafb"

            st.markdown(
                f"""
                <div style='background: {background}; padding: 1rem; border-radius: 0 0 10px 10px;
                border: 2px solid {border_color}; border-top: none; margin-top: -0.5rem; margin-bottom: 1rem;'>
                    <div style='font-size: 0.85rem; color: #6b7280; text-align: center; margin-bottom: 0.5rem;'>
                        {lang_data['script']} Script
                    </div>
                    <div style='font-size: 0.95rem; text-align: center; color: #374151; line-height: 1.5;
                    padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 6px;'>
                        {lang_data['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Only show the rest if a language is selected
    if st.session_state["selected_language"]:
        selected_lang_data = LANGUAGE_EXAMPLES[st.session_state["selected_language"]]

        st.markdown("---")

        # Show the English original for reference
        st.markdown("### 📄 Selected Sentence")
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
            padding: 1.5rem; border-radius: 12px; border: 2px solid #3b82f6; margin-bottom: 1.5rem;'>
                <div style='font-weight: 600; color: #3b82f6; margin-bottom: 0.8rem; font-size: 1.1rem;'>
                    English (Original):
                </div>
                <div style='font-size: 1.15rem; color: #1f2937; line-height: 1.6; margin-bottom: 1rem;'>
                    "{selected_lang_data['english_original']}"
                </div>
                <div style='font-weight: 600; color: #3b82f6; margin-bottom: 0.8rem; font-size: 1.1rem;'>
                    {selected_lang_data['emoji']} {selected_lang_data['language']} ({selected_lang_data['script']} Script):
                </div>
                <div style='font-size: 1.15rem; color: #1f2937; line-height: 1.6;'>
                    "{selected_lang_data['text']}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Context Input Section
        st.markdown("### 🎯 Step 2: Select Translation Context")
        st.markdown(
            "<p style='color: #6b7280; margin-bottom: 1rem;'>Choose the context that will guide the translation. The context will be translated to the source language before being used.</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "⚽ Use Sports Context",
                width='stretch',
                help="Context: This is about a professional soccer match",
            ):
                st.session_state["english_context"] = selected_lang_data["context_sports"]
                st.rerun()

        with col2:
            if st.button(
                "💼 Use Business Context",
                width='stretch',
                help="Context: This is about corporate executives and strategic decisions",
            ):
                st.session_state["english_context"] = selected_lang_data["context_business"]
                st.rerun()

        st.text_area(
            "English Context",
            value=st.session_state.get("english_context", ""),
            height=80,
            help="Provide context to help disambiguate the translation",
            key="english_context",
        )

        st.markdown("---")

        # Translation Actions
        st.markdown("### 🚀 Step 3: Translate")

        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            if st.button("🌐 Translate", width='stretch', type="primary"):
                if not st.session_state["english_context"].strip():
                    st.warning("Please provide context first!")
                else:
                    # Translate without context first
                    with st.spinner("Translating without context..."):
                        st.session_state["translation_without_context"] = translate_with_source_language(
                            st.session_state["selected_text"],
                            st.session_state["selected_language"],
                            "English",
                        )

                    # Then translate the context to source language
                    with st.spinner("Translating context to source language..."):
                        st.session_state["translated_context"] = translate_context_to_language(
                            st.session_state["english_context"],
                            st.session_state["selected_language"],
                        )

                    # Finally translate with context
                    with st.spinner("Translating with context..."):
                        st.session_state["translation_with_context"] = translate_with_source_language(
                            st.session_state["selected_text"],
                            st.session_state["selected_language"],
                            "English",
                            st.session_state["translated_context"],
                        )
                        st.session_state["show_results"] = True

        with col_btn2:
            if st.button("🔄 Clear Results", width='stretch'):
                st.session_state["translated_context"] = ""
                st.session_state["translation_without_context"] = ""
                st.session_state["translation_with_context"] = ""
                st.session_state["show_results"] = False
                st.rerun()

        # Results Section
        if st.session_state["show_results"]:
            st.markdown("---")
            st.markdown("### 📊 Translation Pipeline & Results")

            # Show the translation pipeline
            st.markdown("#### 🔄 Translation Pipeline")

            # Pipeline visualization
            pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

            with pipeline_col1:
                st.markdown(
                    f"""
                    <div style='background: white; padding: 1rem; border-radius: 10px;
                    border: 2px solid #3b82f6; text-align: center;'>
                        <div style='font-weight: 600; color: #3b82f6; margin-bottom: 0.5rem;'>
                            Source Text
                        </div>
                        <div style='font-size: 1.1rem; color: #1f2937;'>
                            {selected_lang_data['emoji']} {st.session_state['selected_language']}
                        </div>
                        <div style='font-size: 0.9rem; color: #374151; margin-top: 0.5rem;
                        padding: 0.5rem; background: #f3f4f6; border-radius: 6px;'>
                            {st.session_state['selected_text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with pipeline_col2:
                if st.session_state["translated_context"]:
                    st.markdown(
                        f"""
                        <div style='background: white; padding: 1rem; border-radius: 10px;
                        border: 2px solid #f59e0b; text-align: center;'>
                            <div style='font-weight: 600; color: #f59e0b; margin-bottom: 0.5rem;'>
                                Context (Translated)
                            </div>
                            <div style='font-size: 0.85rem; color: #6b7280; margin-bottom: 0.3rem;'>
                                English → {st.session_state['selected_language']}
                            </div>
                            <div style='font-size: 0.9rem; color: #374151; margin-top: 0.5rem;
                            padding: 0.5rem; background: #fef3c7; border-radius: 6px;'>
                                {st.session_state['translated_context']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style='background: #f9fafb; padding: 1rem; border-radius: 10px;
                        border: 2px dashed #d1d5db; text-align: center;'>
                            <div style='color: #9ca3af;'>
                                No context provided
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with pipeline_col3:
                st.markdown(
                    """
                    <div style='background: white; padding: 1rem; border-radius: 10px;
                    border: 2px solid #10b981; text-align: center;'>
                        <div style='font-weight: 600; color: #10b981; margin-bottom: 0.5rem;'>
                            Target
                        </div>
                        <div style='font-size: 1.1rem; color: #1f2937;'>
                            🇬🇧 English
                        </div>
                        <div style='font-size: 0.9rem; color: #374151; margin-top: 0.5rem;
                        padding: 0.5rem; background: #d1fae5; border-radius: 6px;'>
                            Translation result →
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison results
            st.markdown("#### 📋 Side-by-Side Comparison")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown(
                    """
                    <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    color: white; border-radius: 10px 10px 0 0; font-weight: 600;'>
                        1️⃣ Without Context
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state["translation_without_context"]:
                    st.markdown(
                        f"""
                        <div style='background: white; padding: 1.5rem; border-radius: 0 0 10px 10px;
                        border: 2px solid #f59e0b; min-height: 150px; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);'>
                            <div style='font-size: 1.2rem; color: #1f2937; line-height: 1.6; font-weight: 500;'>
                                "{st.session_state["translation_without_context"]}"
                            </div>
                            <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
                            color: #6b7280; font-size: 0.9rem;'>
                                ℹ️ Translation without additional context
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style='background: #f9fafb; padding: 1.5rem; border-radius: 0 0 10px 10px;
                        border: 2px dashed #d1d5db; min-height: 150px; display: flex;
                        align-items: center; justify-content: center;'>
                            <div style='color: #9ca3af; text-align: center;'>
                                Click "Translate Without Context" to see results
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with result_col2:
                st.markdown(
                    """
                    <div style='text-align: center; padding: 0.8rem; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                    color: white; border-radius: 10px 10px 0 0; font-weight: 600;'>
                        2️⃣ With Context
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state["translation_with_context"]:
                    # Determine which context was used
                    context_type = ""
                    context_icon = ""
                    if st.session_state["english_context"] == selected_lang_data["context_sports"]:
                        context_type = "Sports"
                        context_icon = "⚽"
                    elif st.session_state["english_context"] == selected_lang_data["context_business"]:
                        context_type = "Business"
                        context_icon = "💼"
                    else:
                        context_type = "Custom"
                        context_icon = "✏️"

                    st.markdown(
                        f"""
                        <div style='background: white; padding: 1.5rem; border-radius: 0 0 10px 10px;
                        border: 2px solid #3b82f6; min-height: 150px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);'>
                            <div style='font-size: 1.2rem; color: #1f2937; line-height: 1.6; font-weight: 500;'>
                                "{st.session_state["translation_with_context"]}"
                            </div>
                            <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
                            color: #6b7280; font-size: 0.9rem;'>
                                ✅ Translation with {context_icon} <strong>{context_type} Context</strong> (translated to {st.session_state['selected_language']})
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style='background: #f9fafb; padding: 1.5rem; border-radius: 0 0 10px 10px;
                        border: 2px dashed #d1d5db; min-height: 150px; display: flex;
                        align-items: center; justify-content: center;'>
                            <div style='color: #9ca3af; text-align: center;'>
                                Click "Translate With Context" to see results
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Show insight if both translations exist
            if (
                st.session_state["translation_without_context"]
                and st.session_state["translation_with_context"]
            ):
                if (
                    st.session_state["translation_without_context"]
                    != st.session_state["translation_with_context"]
                ):
                    st.success(
                        f"✨ **Key Insight:** The translation changed when context was provided! "
                        f"The English context was first translated to {st.session_state['selected_language']}, "
                        f"then used to inform the translation back to English. This demonstrates how transformers "
                        f"use contextual information to resolve ambiguity."
                    )
                else:
                    st.info(
                        "💡 **Observation:** The translations are the same. The model may have "
                        "interpreted the sentence consistently in both cases."
                    )

    # Educational info
    with st.expander("ℹ️ How This Demonstration Works"):
        st.markdown(
            """
            **Translation Pipeline:**

            1. **Source Text Selection**: Choose a sentence in a foreign language (Spanish, Russian, Arabic, Chinese, Japanese, or Greek)
            2. **Context Translation**: Your English context is translated into the source language
            3. **Context-Aware Translation**: The source text is translated to English WITH the translated context
            4. **Comparison**: See how context changes the translation outcome

            **Why This Matters:**

            - **Ambiguity Resolution**: Context-neutral words need domain context for accurate interpretation
            - **Multi-Language Support**: Demonstrates the model can handle diverse scripts (Latin, Cyrillic, Arabic, Hanzi, etc.)
            - **Bidirectional Translation**: Shows the model's ability to translate both TO and FROM English
            - **Context Window**: Illustrates how transformers use additional information to improve translation accuracy
            - **Domain-Specific Terminology**: The same words can mean different things in different domains

            **Example Scenario:**

            The sentence about "teammates", "lead reviewer", "decision", "advisors", and "approach" has dual meanings:

            - **Sports Context**: Professional athletes protesting an offside call during a soccer match
              - Teammates = soccer team players
              - Lead reviewer = head referee/official
              - Decision = ruling on the play
              - Reversed the outcome = overturned the goal
              - Advisors = assistant referees
              - Approach = playing strategy or positioning

            - **Business Context**: Corporate team members objecting to a strategic business decision
              - Teammates = business colleagues/team members
              - Lead reviewer = senior executive/decision maker
              - Decision = strategic choice or recommendation
              - Reversed the outcome = rejected the proposal
              - Advisors = financial consultants/analysts
              - Approach = business strategy or methodology

            By providing context, the model interprets these neutral terms correctly for the intended domain!
            """
        )
