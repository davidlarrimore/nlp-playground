"""Interactive visualization for tokenization and byte-pair encoding."""

import html

import streamlit as st
import streamlit.components.v1 as components

from utils import tokenize_text

# Characters that should always be treated as single-token boundaries.
PUNCTUATION_BREAKS = {".", ",", "!", "?", ";", ":"}


def tokenization_page() -> None:
    """Provide an interactive subword tokenization explorer."""
    # Page header
    st.markdown("# 🔤 Tokenization Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                Watch how GPT-style tokenizers break text into byte-pair tokens.
                Enter any sentence and see both the human-readable token fragments and their numeric IDs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Tokens are the smallest text units a model reads—often subwords or characters—so\n"
        "understanding how a sentence breaks into tokens explains why long words can cost\n"
        "multiple tokens while punctuation may stand alone."
    )

    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_text = "The ancient crystal fortress shuddered dramatically as a newly awakened dragon released an overwhelming wave of uncontrollable magic, forcing a determined band of adventurers to begin an extraordinarily dangerous journey to protect their threatened kingdom."
        text = st.text_area(
            "📝 Enter your text",
            value=sample_text,
            height=120,
            help="Type or paste any text to see how it's tokenized",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        tokens, readable = tokenize_text(text)

        # Enhanced metric display
        st.markdown(
            f"""
            <div style='background: white; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; margin-top: 0.5rem;'>
                <div style='color: #9ca3af; font-size: 0.85rem; text-transform: uppercase;
                letter-spacing: 0.5px; margin-bottom: 0.5rem;'>Token Count</div>
                <div style='color: #3b82f6; font-size: 2.5rem; font-weight: 700;'>{len(tokens)}</div>
                <div style='color: #6b7280; font-size: 0.9rem; margin-top: 0.5rem;'>
                    {len(text)} characters
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not tokens:
        st.info("💡 Start typing to see subword tokens and IDs appear below")
        return

    st.markdown("---")
    st.markdown("### 🏷️ Token Visualization")

    # Build the styled token chips, grouping multi-token words
    def _starts_new_word(index: int, text: str) -> bool:
        """Return True when a token should start a new visual group."""
        if index == 0:
            return True
        trimmed = text.strip()
        if not text:
            return True
        if text[:1].isspace():
            return True
        if trimmed in PUNCTUATION_BREAKS:
            return True
        return False

    # Track contiguous tokens that belong to the same word for nicer chips.
    word_groups = []
    current_group = []

    for idx, (token_id, token_text) in enumerate(zip(tokens, readable)):
        normalized_text = token_text or ""
        if _starts_new_word(idx, normalized_text) and current_group:
            word_groups.append(current_group)
            current_group = []
        current_group.append({"idx": idx, "text": normalized_text})

    if current_group:
        word_groups.append(current_group)

    # Render each group as either a single chip or stacked chips.
    tokens_html = ""
    for group in word_groups:
        is_multi = len(group) > 1
        if is_multi:
            tokens_html += "<span class=\"multi-token-label\">"
        for piece in group:
            safe_token = html.escape(piece["text"])
            tokens_html += f"""
        <span class="token-chip">
            <span class="token-number">#{piece["idx"] + 1}</span>
        <span class="token-text">{safe_token}</span>
        </span>
        """
        if is_multi:
            tokens_html += "</span>"

    style_and_tokens = f"""
    <style>
    .token-wrapper {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        line-height: 2.4;
    }}
    .token-chip {{
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.45rem 0.9rem;
        border-radius: 18px;
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 0.9rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .token-chip:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(59, 130, 246, 0.4);
    }}
    .token-number {{
        background: rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-right: 0.35rem;
    }}
    .token-text {{
        font-weight: 600;
    }}
    .multi-token-label {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #fb923c 0%, #f97316 100%);
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.25);
    }}
    .multi-token-label .token-chip {{
        background: rgba(255, 255, 255, 0.25);
        box-shadow: none;
        color: #0f172a;
        border-radius: 12px;
    }}
    .multi-token-label .token-chip .token-number {{
        background: rgba(255, 255, 255, 0.45);
        color: #0f172a;
    }}
    .multi-token-label .token-chip .token-text {{
        color: #0f172a;
    }}
    </style>
    <div class="token-wrapper">
        {tokens_html}
    </div>
    """
    components.html(
        style_and_tokens,
        height=min(520, max(180, 90 + len(tokens) * 36)),
        scrolling=True,
    )

    # Information section
    st.markdown("---")
    with st.expander("ℹ️ Understanding Tokenization"):
        st.markdown(
            """
            **Key Concepts:**
            - **Subword Tokens**: Words are broken into smaller pieces that can be reused
            - **Token IDs**: Each token has a unique numeric identifier
            - **Case Sensitivity**: Capitalization affects tokenization
            - **Punctuation**: Special characters often become separate tokens

            **Why This Matters:**
            - Token count affects model cost and context limits
            - Understanding tokens helps optimize prompts
            - Different languages may tokenize differently
            """
        )
