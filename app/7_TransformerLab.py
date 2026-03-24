"""Hands-on lab for visualizing transformer sampling and token probabilities."""

import html
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from utils import get_generation_model, get_openai_client, tokenize_text


# Default prompts keep the story consistent across user sessions.
DEFAULT_PROMPT = "Once upon a time, "

DEFAULT_SYSTEM_PROMPT = """You are a master storyteller. Your only job is to continue the user's text naturally, coherently, and in a narrative style."""

# Tokens that should remain glued to the previous word when rendering context.
PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":"}


def format_context_snapshot(text: str, limit: Optional[int] = None) -> str:
    """Return the most recent words from a running context string."""
    cleaned = text.replace("\n", " ").replace("\r", "")
    if not cleaned:
        return ""
    words = cleaned.split()
    if limit is not None:
        words = words[-limit:]
    snapshot = " ".join(words)
    if cleaned and cleaned[-1].isspace():
        if snapshot and not snapshot.endswith(" "):
            snapshot += " "
        elif not snapshot:
            snapshot = " "
    return snapshot


def normalize_token(token: str) -> str:
    """Normalize tokens for context: keep punctuation tight, add trailing spaces."""
    cleaned = token.replace("\n", " ").replace("\r", "")
    cleaned = cleaned if cleaned else " "
    stripped = cleaned.strip()
    if stripped in PUNCTUATION_TOKENS:
        return stripped
    if cleaned.endswith(" "):
        return cleaned
    return f"{cleaned} "


def ensure_min_choices(tokens: List[str], probabilities: np.ndarray, min_count: int = 5) -> Tuple[List[str], np.ndarray]:
    """Pad token list so there are at least `min_count` choices."""
    padded_tokens = tokens.copy()
    padded_probs = probabilities.copy()
    while len(padded_tokens) < min_count:
        padded_tokens.append(" ")
        padded_probs = np.append(padded_probs, 0.0)
    return padded_tokens, padded_probs


def render_token_chips(tokens: List[str], title: str) -> None:
    """Show tokens as pill-shaped chips to mimic the white paper visualization."""
    chips = ""
    for idx, token in enumerate(tokens):
        token_display = token if token.strip() else "<space>"
        token_display = token_display.replace(" ", "")
        safe_token = html.escape(token_display)
        chips += f"""
        <span class="chip">
            <strong>{idx + 1}</strong>
            <span>{safe_token}</span>
        </span>
        """

    html_block = f"""
    <style>
        .chip-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            margin-top: 0.4rem;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.35rem 0.6rem;
            border-radius: 10px;
            background: #eef2ff;
            font-family: "SFMono-Regular", Consolas, monospace;
            font-size: 0.8rem;
            color: #312e81;
        }}
        .chip strong {{
            font-size: 0.7rem;
            color: #7c3aed;
        }}
        .chip span {{
            display: inline-flex;
            align-items: center;
        }}
    </style>
    <div>
        <div style="font-weight: 600; color: #374151; font-size: 0.9rem;">{title}</div>
        <div class="chip-grid">
            {chips}
        </div>
    </div>
    """
    components.html(html_block, height=120, scrolling=False)


def describe_top_choices(
    distribution: Dict[str, float],
    log_probs: Dict[str, float],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Select the highest probability tokens for display.

    IMPORTANT: This function deduplicates tokens to ensure each unique token
    appears only once in the output. Tokens are considered duplicates if they
    have the same text after normalizing (stripping whitespace and lowercasing).
    This prevents "Once" and "once" from both appearing in the results.

    The highest-probability variant of each normalized token is kept.
    """
    # Sort by probability descending
    sorted_tokens = sorted(distribution.items(), key=lambda item: item[1], reverse=True)

    # Deduplicate: track seen tokens and only keep first occurrence (highest prob)
    seen_tokens = set()
    deduplicated = []
    for token, prob in sorted_tokens:
        # Normalize for comparison: strip whitespace AND convert to lowercase
        # This handles "Once" vs "once" vs " Once " as the same token
        token_normalized = token.strip().lower()
        if token_normalized not in seen_tokens:
            seen_tokens.add(token_normalized)
            deduplicated.append((token, prob))
        # else: skip duplicate (already have higher probability version)

    # Take top K after deduplication
    top_choices = deduplicated[:top_k]

    return [
        {
            "token": token,
            "probability": prob,
            "logprob": log_probs.get(token),
        }
        for token, prob in top_choices
    ]


def render_probability_table(top_choices: List[Dict[str, Any]], selected_token: Optional[str] = None) -> None:
    """Display the probability distribution using Streamlit’s native table."""
    rows = []
    for choice in top_choices:
        token = choice["token"].replace(" ", "") or "<space>"
        rows.append(
            {
                "Token": token,
                "Probability": f"{choice['probability']:.2%}",
            }
        )

    dataframe = pd.DataFrame(rows)
    if selected_token is None:
        st.table(dataframe)
        return

    selected_display = selected_token.replace(" ", "") or "<space>"

    def _highlight_selected(row: pd.Series) -> List[str]:
        if row["Token"] == selected_display:
            return ["background-color: #e0e7ff"] * len(row)
        return [""] * len(row)

    styled = dataframe.style.apply(_highlight_selected, axis=1)
    st.dataframe(styled, width='stretch')


def plot_distribution(top_choices: List[Dict[str, Any]]) -> None:
    """Render a bar chart to visualize the sampling distribution."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[choice["token"] for choice in top_choices],
                y=[choice["probability"] for choice in top_choices],
                marker_color="#6366f1",
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Top tokens sampled at the last step",
        yaxis=dict(range=[0, 1], title="Probability", gridcolor="rgba(99,102,241,0.2)"),
        xaxis=dict(title="Token", tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=40, r=20),
        height=250,
    )
    st.plotly_chart(fig, width='stretch')


def request_next_token_distribution(context_text: str, temperature: float, system_prompt: str = DEFAULT_SYSTEM_PROMPT, top_k: int = 8) -> Dict[str, Any]:
    """Call the configured transformer model to get logprob distributions."""
    client = get_openai_client()
    model = get_generation_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": context_text},
        ],
        temperature=temperature,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_k,
    )
    choice = response.choices[0]
    logprobs_obj = getattr(choice, "logprobs", None)

    # Helper to read the assistant message content
    message_obj = getattr(choice, "message", None)
    if message_obj is None and isinstance(choice, dict):
        message_obj = choice.get("message")
    message_content = ""
    if message_obj is not None:
        message_content = getattr(message_obj, "content", "") or (
            message_obj.get("content", "") if isinstance(message_obj, dict) else ""
        )

    if not message_content:
        # Last resort to get the token output
        raw_message = getattr(choice, "content", None)
        if raw_message is None and isinstance(choice, dict):
            raw_message = choice.get("content")
        message_content = raw_message or ""

    # Extract a {token: logprob} mapping from the new-style logprobs API
    top_logprobs: Dict[str, float] = {}
    if logprobs_obj is not None:
        content_list = getattr(logprobs_obj, "content", None)
        if content_list:
            first_item = content_list[0]
            candidates = getattr(first_item, "top_logprobs", None)
            if candidates:
                for cand in candidates:
                    token = (
                        getattr(cand, "token", None)
                        or (cand.get("token") if isinstance(cand, dict) else None)
                    )
                    logprob = (
                        getattr(cand, "logprob", None)
                        or (cand.get("logprob") if isinstance(cand, dict) else None)
                    )
                    if token is not None and logprob is not None:
                        top_logprobs[token] = float(logprob)

    # Fallback for older / unexpected response shapes
    if not top_logprobs and logprobs_obj is not None and getattr(logprobs_obj, "top_logprobs", None):
        raw_top = logprobs_obj.top_logprobs[0]
        if isinstance(raw_top, dict):
            top_logprobs = {k: float(v) for k, v in raw_top.items()}

    # Deduplicate tokens by keeping only the highest logprob for each unique token
    # This handles cases where the API returns the same token multiple times
    deduplicated_logprobs: Dict[str, float] = {}
    for token, logprob in top_logprobs.items():
        if token not in deduplicated_logprobs or logprob > deduplicated_logprobs[token]:
            deduplicated_logprobs[token] = logprob

    tokens = list(deduplicated_logprobs.keys())
    if not tokens:
        tokens = [message_content.strip() or " "]
        deduplicated_logprobs = {tokens[0]: 0.0}

    logps = np.array([float(deduplicated_logprobs[token]) for token in tokens], dtype=float)
    # Convert log probabilities into a numerically stable probability simplex.
    logps = logps - np.max(logps)
    probs = np.exp(logps)
    total = np.sum(probs)
    if total <= 0 or probs.size == 0:
        normalized = np.ones_like(probs) / max(probs.size, 1)
    else:
        normalized = probs / total
    if normalized.size == 0:
        normalized = np.array([1.0], dtype=float)
    # Normalize spacing so displayed tokens look natural inside the UI.
    # CRITICAL: We must deduplicate AFTER normalization because different tokens
    # (e.g., "Once" vs " Once") may normalize to the same value ("Once ")
    norm_token_to_prob: Dict[str, float] = {}
    for token, prob in zip(tokens, normalized):
        norm_token = normalize_token(token)
        # Keep the highest probability if we see the same normalized token multiple times
        if norm_token not in norm_token_to_prob or prob > norm_token_to_prob[norm_token]:
            norm_token_to_prob[norm_token] = float(prob)

    # Build the final lists from deduplicated data
    norm_tokens = list(norm_token_to_prob.keys())
    normalized = np.array([norm_token_to_prob[t] for t in norm_tokens], dtype=float)

    # Renormalize probabilities after deduplication
    total = np.sum(normalized)
    if total > 0:
        normalized = normalized / total

    raw_log_probs = {
        normalize_token(token): float(logprob)
        for token, logprob in deduplicated_logprobs.items()
    }

    norm_tokens, normalized = ensure_min_choices(norm_tokens, normalized)
    selected_idx = int(np.argmax(normalized))
    distribution = dict(zip(norm_tokens, normalized))

    # Capture the raw API response in a JSON-serializable form for debugging/teaching
    try:
        raw_response = response.model_dump()
    except Exception:
        try:
            raw_response = response.to_dict()
        except Exception:
            raw_response = str(response)

    return {
        "token": norm_tokens[selected_idx],
        "distribution": distribution,
        "raw_response": raw_response,
        "log_probs": raw_log_probs,
        "message_content": message_content,
    }




def transformer_page() -> None:
    """Render the transformer white paper inspired demo page."""
    # Add global CSS to prevent vertical scrolling
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
                max-height: 100vh;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("# ⚡ Transformer Lab")

    # Introduction section
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2);'>
            <div style='color: white;'>
                <h3 style='margin-top: 0; color: white;'>🎯 What Is This?</h3>
                <p style='font-size: 1.05rem; line-height: 1.6; margin-bottom: 1rem;'>
                    This interactive lab demonstrates how transformer models generate text one token at a time through <strong>autoregressive sampling</strong>. Watch as the model predicts the most likely next token based on context, revealing the probabilistic nature of AI text generation.
                </p>
                <h3 style='color: white;'>📚 What You'll Learn</h3>
                <ul style='font-size: 1rem; line-height: 1.8; margin-bottom: 1rem;'>
                    <li><strong>Token-by-Token Generation:</strong> How transformers build text incrementally, choosing one token at a time</li>
                    <li><strong>Probability Distributions:</strong> See the likelihood scores for each potential next token</li>
                    <li><strong>Temperature Effects:</strong> Understand how temperature controls randomness vs. determinism</li>
                    <li><strong>Context Impact:</strong> Observe how previous tokens influence future predictions</li>
                    <li><strong>Alternative Paths:</strong> Explore how different token choices create different stories</li>
                </ul>
                <h3 style='color: white;'>🎮 How to Use This Lab</h3>
                <ol style='font-size: 1rem; line-height: 1.8; margin-bottom: 0.5rem;'>
                    <li><strong>Set Your Prompt:</strong> Enter a starting phrase or use the default story prompt</li>
                    <li><strong>Adjust Temperature:</strong> Lower (0.2) = predictable, Higher (1.0) = creative and random</li>
                    <li><strong>Generate Tokens:</strong> Click "Generate First Token" to start, then "Generate Next Token" to continue</li>
                    <li><strong>Explore Alternatives:</strong> Click any token in the probability table to rewrite the story from that point</li>
                    <li><strong>Navigate History:</strong> Use Previous/Next buttons to review past token choices</li>
                </ol>
                <div style='background: rgba(255, 255, 255, 0.15); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <strong>💡 Pro Tip:</strong> Try the same prompt with different temperatures to see how creativity changes the output!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Transformers process every token position in parallel and use attention to focus on the most relevant words "
        "when predicting the next token—unlike RNNs/LSTMs that march through text sequentially. As you step through "
        "generation below, notice how the probability table shifts based on the full context window."
    )

    # Metadata for the three high-level cards displayed on the page.
    panel_meta = {
        "Prompt Setup": {
            "desc": "Set the scene with a starter prompt, temperature, and step count.",
            "tip": "Slide 1 · Kick off the story for the lunch & learn.",
            "bg": "linear-gradient(135deg, #eef2ff 0%, #c7d2fe 100%)",
            "border": "#c4b5fd",
        },
        "Sampling Journey": {
            "desc": "Step through each autoregressive choice and understand the confidence profile.",
            "tip": "Slide 2 · Watch sampling play out like a live demo.",
            "bg": "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
            "border": "#fcd34d",
        },
        "Final Output": {
            "desc": "Celebrate the final story and the transformer concepts that made it possible.",
            "tip": "Slide 3 · Wrap up with insights for lunch attendees.",
            "bg": "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)",
            "border": "#86efac",
        },
    }

    # Initialize session state with sensible defaults for first-time visitors.
    if "prompt_source" not in st.session_state:
        st.session_state["prompt_source"] = DEFAULT_PROMPT
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.8
    if "show_detail" not in st.session_state:
        st.session_state["show_detail"] = True
    if "token_history" not in st.session_state:
        st.session_state["token_history"] = []
    if "current_context" not in st.session_state:
        st.session_state["current_context"] = ""
    if "viewing_index" not in st.session_state:
        st.session_state["viewing_index"] = 0
    if "prompt_tokens" not in st.session_state:
        st.session_state["prompt_tokens"] = []

    # Local references keep the rendering code readable.
    token_history = st.session_state["token_history"]
    viewing_index = st.session_state["viewing_index"]
    is_setup = len(token_history) == 0

    if is_setup:
        active_panel = panel_meta["Prompt Setup"]
        card_desc = active_panel["desc"]
        card_tip = active_panel["tip"]
        card_bg = active_panel["bg"]
        card_border = active_panel["border"]
    else:
        step = token_history[viewing_index]
        active_panel = panel_meta["Sampling Journey"].copy()
        card_desc = f"Token #{viewing_index + 1} selected: {step['selected'].strip() or '<space>'}."
        card_tip = f"Response {viewing_index + 1} of {len(token_history)} · Showing probability distribution."
        card_bg = active_panel["bg"]
        card_border = active_panel["border"]

    # Display the growing story at the top (if we're past setup)
    if not is_setup:
        current_story = st.session_state["current_context"]
        # Build the story with color coding
        # - Starter prompt = white
        # - Previously added tokens = yellow
        # - Current token at viewing_index = red

        story_html = f'<span style="color: white;">{html.escape(st.session_state["prompt_source"])}</span>'

        for i in range(viewing_index + 1):
            if i < len(token_history):
                token_text = html.escape(token_history[i]["selected"])
                if i == viewing_index:
                    # Current token is red
                    story_html += f'<span style="color: #ef4444;">{token_text}</span>'
                else:
                    # Previously added tokens are yellow
                    story_html += f'<span style="color: #fbbf24;">{token_text}</span>'

        at_latest = viewing_index == len(token_history) - 1
        status_badge = "🟢 Latest" if at_latest else f"⏮️ Viewing History"

        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1.5rem; border-radius: 14px; margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);'>
                <div style='font-size: 2rem; font-weight: 700; line-height: 1.4;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    {story_html}
                </div>
                <div style='margin-top: 0.75rem; display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-size: 0.9rem; color: #bfdbfe; font-weight: 500;'>
                        📝 Token #{viewing_index + 1} of {len(token_history)} Generated
                    </div>
                    <div style='background: {"#10b981" if at_latest else "#f59e0b"}; color: white;
                    padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;'>
                        {status_badge}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    panel_card = f"""
    <style>
        .panel-card {{
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            background: {card_bg};
            border: 1px solid {card_border};
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }}
        .panel-card .panel-subtitle {{
            margin: 0.1rem 0;
            font-size: 0.9rem;
            color: #1f2937;
            font-weight: 600;
        }}
        .panel-card .panel-tip {{
            margin: 0.2rem 0 0;
            font-size: 0.75rem;
            color: #4b5563;
        }}
    </style>
    """
    st.markdown(panel_card, unsafe_allow_html=True)

    prompt_text = st.session_state["prompt_source"]
    context_prompt = prompt_text if prompt_text.strip() else DEFAULT_PROMPT

    if is_setup:
        col_main, col_side = st.columns([2.5, 1.5])
        with col_main:
            st.text_area(
                "🤖 System Prompt",
                value=st.session_state["system_prompt"],
                key="system_prompt",
                height=120,
                help="Instructions that guide the model's behavior for token generation",
            )
            st.text_area(
                "✍️ Enter a starter prompt",
                value=prompt_text,
                key="prompt_source",
                height=120,
            )
            temperature = st.slider(
                "Temperature (creativity)",
                min_value=0.2,
                max_value=1.0,
                value=st.session_state["temperature"],
                step=0.1,
                key="temperature",
                help="Higher temperature = more creative/random token selection",
            )
            _ = st.checkbox(
                "Show detailed API response",
                value=st.session_state["show_detail"],
                key="show_detail",
            )
        with col_side:
            _, prompt_tokens = tokenize_text(context_prompt)
            st.markdown("#### ⚡ Prompt Metrics")
            st.metric("Token count", len(prompt_tokens))
            render_token_chips(prompt_tokens[:8], "First 8 tokens")
            st.markdown(
                """
                <div style='background: #f0f9ff; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; margin-top: 1rem;'>
                <strong>🚦 How It Works</strong>
                <ul style='margin: 0.3rem 0 0; padding-left: 1.2rem;'>
                <li>Each "Next" generates one token</li>
                <li>Click any token to rewrite the story</li>
                <li>Temperature controls creativity</li>
                <li>Navigate back to see history</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        # Viewing a token from history
        detail_mode = st.session_state["show_detail"]
        step = token_history[viewing_index]

        # Show rewrite notification if it exists
        if "rewrite_notification" in st.session_state:
            st.info(st.session_state["rewrite_notification"])
            del st.session_state["rewrite_notification"]

        # Probability table on left, token selection + API response on right
        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            st.markdown("### 📊 Probability Distribution")

            # Show warning if not at latest token
            at_latest = viewing_index == len(token_history) - 1
            if not at_latest:
                tokens_ahead = len(token_history) - viewing_index - 1
                st.warning(f"⚠️ You're viewing history. Selecting a different token will remove {tokens_ahead} future token(s).")
            else:
                st.caption("💡 Select a different token to rewrite the story from this point. Click any row below to choose a different token.")

            # Build the probability table using Streamlit-native components
            table_rows = []
            current_selected_idx = 0
            selected_reference = step.get("message_content_normalized", step["selected"])
            for idx, choice in enumerate(step["top_choices"]):
                token_display = choice["token"].replace(" ", "") or "<space>"
                logprob_value = choice.get("logprob")
                if logprob_value is not None:
                    prob_display = f"{np.exp(logprob_value):.5%}"
                    logprob_display = f"{logprob_value:.5f}"
                else:
                    prob_display = f"{choice['probability']:.5%}"
                    logprob_display = ""
                table_rows.append(
                    {
                        "Token": token_display,
                        "Probability": prob_display,
                        "Log Probability": logprob_display,
                        "token_obj": choice,
                        "row_idx": idx,
                    }
                )
                if choice["token"] == selected_reference:
                    current_selected_idx = idx

            selection_key = f"token_sel_{viewing_index}"
            if selection_key not in st.session_state:
                st.session_state[selection_key] = current_selected_idx

            if table_rows:
                prob_df = pd.DataFrame(
                    [
                        {
                            "Token": row["Token"],
                            "Probability": row["Probability"],
                            "Log Probability": row["Log Probability"],
                            "row_idx": row["row_idx"],
                        }
                        for row in table_rows
                    ]
                )

                st.session_state[selection_key] = min(
                    st.session_state[selection_key], len(table_rows) - 1
                )

                gb = GridOptionsBuilder.from_dataframe(prob_df)
                gb.configure_column("row_idx", hide=True)
                gb.configure_selection(
                    selection_mode="single",
                    use_checkbox=False,
                    suppressRowClickSelection=False,
                    pre_selected_rows=[st.session_state[selection_key]],
                )
                grid_options = gb.build()
                highlight_js = JsCode(
                    """
                    function(params) {
                        return params.data.row_idx === %d
                            ? {backgroundColor: '#e0e7ff'}
                            : null;
                    }
                    """
                    % st.session_state[selection_key]
                )
                grid_options["getRowStyle"] = highlight_js

                grid_response = AgGrid(
                    prob_df,
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=True,
                    fit_columns_on_grid_load=True,
                    theme="alpine",
                    height=300,
                    key=f"{selection_key}_aggrid",
                )

                selected_rows = grid_response.get("selected_rows")
                if isinstance(selected_rows, pd.DataFrame):
                    selected_rows = selected_rows.to_dict("records")
                if not selected_rows:
                    selected_rows = []
                if selected_rows:
                    selected_candidate = int(selected_rows[0]["row_idx"])
                    if 0 <= selected_candidate < len(table_rows):
                        st.session_state[selection_key] = selected_candidate

                selected_idx = st.session_state[selection_key]
                selected_choice = table_rows[selected_idx]["token_obj"]
            else:
                selected_choice = {"token": step["selected"], "probability": step.get("prob", 0.0)}

            # Check if selection changed
            if selected_choice["token"] != step["selected"]:
                # Replace the token at this position
                new_token = selected_choice["token"]

                # Rebuild context up to this point
                rebuilt_context = st.session_state["prompt_source"]
                for i in range(viewing_index):
                    if i < len(token_history):
                        rebuilt_context += token_history[i]["selected"]

                # Add spacing logic for new token
                if rebuilt_context and not rebuilt_context[-1].isspace():
                    if new_token:
                        first_char = new_token[0]
                        if not first_char.isspace() and first_char not in PUNCTUATION_TOKENS:
                            rebuilt_context += " "

                rebuilt_context += new_token

                # Update the token at this position
                st.session_state["token_history"][viewing_index]["selected"] = new_token
                st.session_state["token_history"][viewing_index]["prob"] = selected_choice["probability"]

                # Delete all tokens after this point (story has diverged)
                tokens_deleted = len(st.session_state["token_history"]) - viewing_index - 1
                st.session_state["token_history"] = st.session_state["token_history"][:viewing_index + 1]

                # Update current context
                st.session_state["current_context"] = rebuilt_context

                # Update context snapshot
                st.session_state["token_history"][viewing_index]["context_snapshot"] = format_context_snapshot(rebuilt_context, limit=12)

                # Store notification for user
                if tokens_deleted > 0:
                    st.session_state["rewrite_notification"] = f"🔄 Story rewritten! {tokens_deleted} future token(s) removed."

                st.rerun()

        with col_right:
            st.markdown("### 🔁 Token Selection")
            st.markdown(
                f"""
                <div style='background: #f0f9ff; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
                    <div style='font-size: 0.85rem; color: #0369a1;'><strong>Selected Token:</strong> {html.escape(step['selected'].strip() or '<space>')}</div>
                    <div style='font-size: 0.85rem; color: #0369a1;'><strong>Probability:</strong> {step['prob']:.2%}</div>
                    <div style='font-size: 0.85rem; color: #0369a1; margin-top: 0.3rem;'><strong>Context Snapshot:</strong> {html.escape(step['context_snapshot'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Educational info about token vocabulary and capitalization
            st.markdown(
                """
                <div style='background: #fef3c7; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; margin-bottom: 1rem; border-left: 4px solid #f59e0b;'>
                    <strong>💡 Did You Know?</strong><br/>
                    Language models have <strong>separate tokens</strong> for different capitalizations!
                    For example, "Once" (capitalized) and "once" (lowercase) are distinct vocabulary entries
                    because capitalization carries meaning in language (sentence beginnings, proper nouns, emphasis).
                    <br/><br/>
                    <strong>What you see here:</strong> We've deduplicated case variants to avoid confusion,
                    showing only the <strong>highest-probability version</strong> of each word. The model
                    actually considers all variants when generating text!
                </div>
                """,
                unsafe_allow_html=True,
            )
    # Navigation footer
    st.markdown("<br>", unsafe_allow_html=True)

    if is_setup:
        # Only show "Start Generating" button
        if st.button("🚀 Generate First Token", width='stretch', type="primary", key="start_gen"):
            # Generate the first token
            try:
                prompt = st.session_state["prompt_source"] if st.session_state["prompt_source"].strip() else DEFAULT_PROMPT
                temperature = st.session_state["temperature"]

                # Initialize context
                st.session_state["current_context"] = prompt
                _, prompt_tokens = tokenize_text(prompt)
                st.session_state["prompt_tokens"] = prompt_tokens

                # Get first token
                system_prompt = st.session_state["system_prompt"]
                token_data = request_next_token_distribution(prompt, temperature, system_prompt)
                next_token = token_data["token"]
                distribution = token_data["distribution"]
                log_probs = token_data.get("log_probs", {})
                message_content = token_data.get("message_content", next_token)
                message_content_norm = normalize_token(message_content)

                # Add spacing logic
                if prompt and not prompt[-1].isspace():
                    if next_token:
                        first_char = next_token[0]
                        if not first_char.isspace() and first_char not in PUNCTUATION_TOKENS:
                            st.session_state["current_context"] += " "

                st.session_state["current_context"] += next_token

                top_choices = describe_top_choices(distribution, log_probs)
                st.session_state["token_history"].append({
                    "step": 1,
                    "selected": next_token,
                    "prob": float(distribution.get(next_token, 0.0)),
                    "message_content": message_content,
                    "message_content_normalized": message_content_norm,
                    "top_choices": top_choices,
                    "context_snapshot": format_context_snapshot(st.session_state["current_context"], limit=12),
                    "raw_response": token_data.get("raw_response"),
                })
                st.session_state["viewing_index"] = 0
                st.rerun()
            except Exception as exc:
                st.error(f"API error: {exc}")
    else:
        nav_cols = st.columns([1, 1])
        can_prev = viewing_index > 0
        at_latest = viewing_index == len(token_history) - 1

        with nav_cols[0]:
            if st.button("⬅️ Previous Token", width='stretch', disabled=not can_prev, key="nav_prev"):
                st.session_state["viewing_index"] = viewing_index - 1
                st.rerun()

        with nav_cols[1]:
            if at_latest:
                # Generate next token
                if st.button("➡️ Generate Next Token", width='stretch', type="primary", key="gen_next"):
                    try:
                        temperature = st.session_state["temperature"]
                        context = st.session_state["current_context"]

                        # Get next token
                        system_prompt = st.session_state["system_prompt"]
                        token_data = request_next_token_distribution(context, temperature, system_prompt)
                        next_token = token_data["token"]
                        distribution = token_data["distribution"]
                        log_probs = token_data.get("log_probs", {})
                        message_content = token_data.get("message_content", next_token)
                        message_content_norm = normalize_token(message_content)

                        # Add spacing logic
                        if context and not context[-1].isspace():
                            if next_token:
                                first_char = next_token[0]
                                if not first_char.isspace() and first_char not in PUNCTUATION_TOKENS:
                                    st.session_state["current_context"] += " "

                        st.session_state["current_context"] += next_token

                        top_choices = describe_top_choices(distribution, log_probs)
                        st.session_state["token_history"].append({
                            "step": len(token_history) + 1,
                            "selected": next_token,
                            "prob": float(distribution.get(next_token, 0.0)),
                            "message_content": message_content,
                            "message_content_normalized": message_content_norm,
                            "top_choices": top_choices,
                            "context_snapshot": format_context_snapshot(st.session_state["current_context"], limit=12),
                            "raw_response": token_data.get("raw_response"),
                        })
                        st.session_state["viewing_index"] = len(st.session_state["token_history"]) - 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"API error: {exc}")
            else:
                # Navigate forward in history
                if st.button("Next Token ➡️", width='stretch', type="primary", key="nav_next"):
                    st.session_state["viewing_index"] = viewing_index + 1
                    st.rerun()

        # Add reset button
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset & Start New Story", width='stretch', key="reset"):
            st.session_state["token_history"] = []
            st.session_state["current_context"] = ""
            st.session_state["viewing_index"] = 0
            st.session_state["prompt_tokens"] = []
            st.session_state["prompt_source"] = DEFAULT_PROMPT
            st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
            st.rerun()

    # API Response section at the very bottom (only when viewing history)
    if not is_setup and st.session_state.get("show_detail", True):
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔍 API Response Details")
        with st.expander("View raw JSON response", expanded=False):
            step = token_history[viewing_index]
            st.json(step.get("raw_response") or {})
