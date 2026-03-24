"""Homepage and navigation shell for the NLP Playground."""

import streamlit as st

import importlib.util
import sys
from pathlib import Path

# Dynamically import from renamed modules
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the directory where this file is located
current_dir = Path(__file__).parent

# Import the page functions from renamed files using relative paths
sent_module = import_from_file("2_SentimentAnalysisLab", current_dir / "2_SentimentAnalysisLab.py")
trans_module = import_from_file("3_TranslationContextLab", current_dir / "3_TranslationContextLab.py")
bleu_module = import_from_file("4_BLEUEvalLab", current_dir / "4_BLEUEvalLab.py")
tok_module = import_from_file("5_TokenizationLab", current_dir / "5_TokenizationLab.py")
emb_module = import_from_file("6_EmbeddingsLab", current_dir / "6_EmbeddingsLab.py")
tf_module = import_from_file("7_TransformerLab", current_dir / "7_TransformerLab.py")


sentiment_analysis_page = sent_module.sentiment_analysis_page
translation_page = trans_module.translation_page
bleu_page = bleu_module.bleu_page
tokenization_page = tok_module.tokenization_page
embeddings_page = emb_module.embeddings_page
transformer_page = tf_module.transformer_page



# Primary navigation map used by the sidebar radio options.
PAGE_NAVIGATION = {
    "🏠 Home": ("Home / Overview", "🏠"),
    "💭 Sentiment Analysis Lab": ("Sentiment & Topic Modeling Lab", "💭"),
    "🌐 Translation Lab": ("Translation Sandbox Lab", "🌐"),
    "🔤 Tokenization Lab": ("Tokenization Playground Lab", "🔤"),
    "🧮 BLEU Evaluation Lab": ("Translation Quality (BLEU) Lab", "🧮"),
    "🎯 Embeddings Lab": ("Embedding Similarity Lab", "🎯"),
    "⚡ Transformer Lab": ("Transformer Insight Lab", "⚡"),
}


def apply_custom_css() -> None:
    """Apply custom CSS for modern, responsive design."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main {
            padding: 2rem 1rem;
        }

        /* Custom card styling */
        .custom-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            color: white;
        }

        .custom-card h1 {
            color: white;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .custom-card p {
            font-size: 1.1rem;
            opacity: 0.95;
        }

        /* Feature card styling */
        .feature-card {
            background: white;
            border: 2px solid #f0f2f6;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
            border-color: #667eea;
        }

        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }

        .feature-desc {
            color: #6b7280;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Info box styling */
        .info-box {
            background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }

        .info-box h3 {
            color: #667eea;
            margin-bottom: 0.8rem;
            font-size: 1.3rem;
        }

        /* Metric styling */
        .custom-metric {
            background: white;
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        /* Button enhancements */
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }

        /* Navigation link styling */
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"][aria-hidden="true"] {
            display: none;
        }

        [data-testid="stSidebar"] .stRadio > div {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        [data-testid="stSidebar"] .stRadio label {
            background: rgba(255, 255, 255, 0.08);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            transition: all 0.3s ease;
            color: white;
            display: flex;
            align-items: center;
            width: 100%;
            box-sizing: border-box;
            cursor: pointer;
            font-weight: 600;
            font-size: 1.05rem;
            gap: 0.5rem;
        }

        [data-testid="stSidebar"] .stRadio label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] .stRadio label > div:last-child {
            flex: 1;
        }

        [data-testid="stSidebar"] .stRadio label > div:last-child * {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: block;
        }

        [data-testid="stSidebar"] .stRadio label[aria-checked="true"] {
            background: rgba(255, 255, 255, 0.25);
            color: #ffffff;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.18);
            color: #ffffff;
            transform: translateX(4px);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .custom-card h1 {
                font-size: 1.8rem;
            }

            .custom-card p {
                font-size: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_home() -> None:
    """Render the landing overview with objectives and instructions."""
    # Hero section
    st.markdown(
        """
        <div class="custom-card">
            <h1>🎮 NLP Playground</h1>
            <p>A hands-on learning environment for exploring how modern AI systems understand and process human language</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Welcome and orientation section
    st.markdown("## 👋 Welcome Students!")
    st.markdown("""
    This interactive playground lets you explore the fundamental concepts behind natural language processing (NLP)
    and large language models. Each demonstration is designed to help you understand **how** AI systems work,
    not just **what** they do.
    """)

    st.divider()

    # What you'll learn section
    st.markdown("## 📚 What You'll Learn")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Core NLP Concepts
        - **Text Tokenization**: How AI breaks down language into processable units
        - **Vector Embeddings**: How meaning is represented mathematically
        - **Semantic Similarity**: How AI measures the closeness of ideas
        """)

    with col2:
        st.markdown("""
        #### Real-World Applications
        - **Machine Translation**: Context-aware language translation
        - **Sentiment Analysis**: Understanding emotion and opinion in text
        - **Attention Mechanisms**: How transformers focus on relevant information
        """)

    st.divider()

    # How to navigate section
    st.markdown("## 🧭 How to Navigate")

    st.info("""
    **Use the sidebar menu on the left** to explore different demonstrations. Each page is self-contained
    and interactive—you can experiment with your own text and see results in real-time.

    💡 **Tip**: Start with **Tokenization Lab** to understand the basics, then progress through the other topics in order.
    """)

    st.divider()

    # Interactive learning modules
    st.markdown("## 🎯 Learning Modules")

    # Feature cards in columns with clickable navigation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔤 Tokenization Lab")
        st.markdown("""
        **What you'll discover**: How AI models break text into tokens (the basic units of processing)

        **Learning objectives**:
        - Understand byte-pair encoding (BPE)
        - See how different words use different numbers of tokens
        - Learn why token count matters for AI systems
        """)
        if st.button("Start Tokenization Lab →", use_container_width=True, key="tokenization"):
            st.session_state.page = "🔤 Tokenization Lab"
            st.rerun()
        st.markdown("")

        st.markdown("### 🌐 Translation Lab")
        st.markdown("""
        **What you'll discover**: How context improves machine translation accuracy

        **Learning objectives**:
        - Compare translations with and without context
        - Understand ambiguity in language
        - Explore multi-language support
        """)
        if st.button("Start Translation Lab →", use_container_width=True, key="translation"):
            st.session_state.page = "🌐 Translation Lab"
            st.rerun()
        st.markdown("")

        st.markdown("### ⚡ Transformer Lab")
        st.markdown("""
        **What you'll discover**: How attention mechanisms help AI understand context

        **Learning objectives**:
        - Visualize how transformers process text
        - Understand attention patterns
        - Explore prompt engineering effects
        """)
        if st.button("Start Transformer Lab →", use_container_width=True, key="transformer"):
            st.session_state.page = "⚡ Transformer Lab"
            st.rerun()

    with col2:
        st.markdown("### 🎯 Embeddings Lab")
        st.markdown("""
        **What you'll discover**: How AI represents meaning as high-dimensional vectors

        **Learning objectives**:
        - Visualize semantic similarity between texts
        - Understand cosine similarity calculations
        - See how embeddings capture meaning
        """)
        if st.button("Start Embeddings Lab →", use_container_width=True, key="embeddings"):
            st.session_state.page = "🎯 Embeddings Lab"
            st.rerun()
        st.markdown("")

        st.markdown("### 💭 Sentiment Analysis Lab")
        st.markdown("""
        **What you'll discover**: How traditional NLP techniques analyze emotion and extract topics

        **Learning objectives**:
        - Understand VADER sentiment scoring
        - Explore topic modeling with LDA and NMF
        - Analyze real customer reviews
        """)
        if st.button("Start Sentiment Lab →", use_container_width=True, key="sentiment"):
            st.session_state.page = "💭 Sentiment Analysis Lab"
            st.rerun()

    st.divider()

    # Getting started tips
    st.markdown("## 🚀 Getting Started Tips")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        #### 💡 Best Learning Approach
        1. **Start with Tokenization Lab** to understand the basics
        2. **Move to Embeddings Lab** to see how meaning is captured
        3. **Try Translation Lab** to see context in action
        4. **Explore Sentiment Analysis Lab** for practical applications
        5. **Finish with Transformer Lab** to see the full picture
        """)

    with col_right:
        st.markdown("""
        #### 🎮 Interactive Features
        - **Experiment freely**: Try your own text in each demo
        - **Adjust parameters**: Use sliders and controls to see how results change
        - **Read explanations**: Each page includes tooltips and descriptions
        - **Observe patterns**: Look for trends in how the AI responds
        """)

    st.success("✅ All demonstrations run in real-time. Your input is processed immediately, so feel free to experiment!")


def main() -> None:
    """Route to the selected page."""
    st.set_page_config(
        page_title="NLP Playground",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    apply_custom_css()

    # Sidebar navigation with title and icons
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
            <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🎮 NLP Playground</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"

    nav_options = list(PAGE_NAVIGATION.keys())
    # Use session state to set the index of the radio button
    try:
        current_index = nav_options.index(st.session_state.page)
    except ValueError:
        current_index = 0
        st.session_state.page = "🏠 Home"

    selection = st.sidebar.radio("Navigation", nav_options, index=current_index, label_visibility="collapsed")

    # Update session state when sidebar is used
    if selection != st.session_state.page:
        st.session_state.page = selection

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem; color: white;'>
            <p style='font-size: 0.9rem; opacity: 0.8;'>🤖 Powered by Amivero Labs</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to pages using session state
    if st.session_state.page == "🏠 Home":
        show_home()
    elif st.session_state.page == "💭 Sentiment Analysis Lab":
        sentiment_analysis_page()
    elif st.session_state.page == "🌐 Translation Lab":
        translation_page()
    elif st.session_state.page == "🔤 Tokenization Lab":
        tokenization_page()
    elif st.session_state.page == "🎯 Embeddings Lab":
        embeddings_page()
    elif st.session_state.page == "⚡ Transformer Lab":
        transformer_page()
    elif st.session_state.page == "🧮 BLEU Evaluation Lab":
        bleu_page()


if __name__ == "__main__":
    main()
