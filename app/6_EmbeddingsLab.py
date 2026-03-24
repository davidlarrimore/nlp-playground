"""Semantic search playground that visualizes embedding similarity."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import embed_text, cosine_similarity


# Fictional reports used as the embedding search corpus.
DATABASE_ENTRIES = [
    {
        "title": "Shoplifting",
        "text": "[2024-01-15 09:45] Case #3012: Shoplifting incident at Walmart - two juveniles observed concealing electronics and clothing items, fled on foot when confronted by security."
    },
    {
        "title": "Assault",
        "text": "[2024-01-16 18:30] Incident #3156: Aggravated assault reported at Murphy's Bar, victim sustained lacerations to face and torso, suspect described as 6'2\" male wearing leather jacket."
    },
    {
        "title": "Vehicle Pursuit",
        "text": "[2024-01-17 14:23] Report #2891: Vehicle pursuit terminated after suspect in maroon pickup truck evaded officers at high speed on Highway 101 northbound."
    },
    {
        "title": "Domestic Disturbance",
        "text": "[2024-01-18 11:20] Report #3298: Domestic disturbance at 1425 Oak Street, neighbors reported loud arguing and breaking glass, both parties separated upon arrival."
    },
    {
        "title": "Burglary",
        "text": "[2024-01-19 22:15] Case #3445: Burglary in progress at residential address, suspect gained entry through rear window, homeowner's security system triggered alarm at 10:15 PM."
    },
    {
        "title": "Construction Theft",
        "text": "[2024-01-20 16:40] Report #3567: Theft reported from construction site - power tools and copper wiring stolen overnight, estimated value $8,000."
    }
]


def describe_similarity(score: float) -> str:
    """Provide a short explanation based on similarity thresholds."""
    if score >= 0.9:
        return "🟢 Very High Similarity - Nearly identical meanings"
    if score >= 0.75:
        return "🟡 High Similarity - Related concepts and topics"
    if score >= 0.5:
        return "🟠 Moderate Similarity - Some common themes"
    return "🔴 Low Similarity - Distant or different topics"


def get_similarity_color(score: float) -> str:
    """Return a color based on similarity score."""
    if score >= 0.9:
        return "#10b981"  # green
    if score >= 0.75:
        return "#f59e0b"  # yellow
    if score >= 0.5:
        return "#f97316"  # orange
    return "#ef4444"  # red


def embeddings_page() -> None:
    """Show embeddings and cosine similarity in an interactive chart."""
    st.markdown("# 🎯 Embeddings Lab")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;'>
            <p style='margin: 0; color: #4b5563; font-size: 1.05rem;'>
                <strong>How it works:</strong> Vector embeddings convert text into numerical representations that capture
                meaning. When you search for a concept, the system finds reports with similar <em>meaning</em>, not just
                matching keywords.
            </p>
            <p style='margin: 0.8rem 0 0 0; color: #4b5563;'>
                <strong>Try this:</strong> Search for "red truck fleeing the scene" and see how it matches the vehicle pursuit
                report, while showing low similarity to unrelated incidents like shoplifting or domestic disturbances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Embeddings place semantically similar words close together in vector space—unlike "
        "one-hot vectors that only signal presence, these dense numbers capture meaning and "
        "relationships such as king ≈ queen and man → woman."
    )

    # Initialize search query in session state
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = "Looking for incidents with a red truck fleeing the scene"

    # Overview of the vector collection
    st.markdown("### 📋 Vector Collection Overview")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 4px solid #3b82f6;'>
            <p style='margin: 0 0 0.8rem 0; color: #1e3a8a; font-size: 1.05rem;'>
                <strong>What's in this collection?</strong> This vector database contains <strong>individual police report records</strong>
                that have been converted from plain text into numerical vectors (embeddings). Each report captures details about
                incidents like shoplifting, assaults, vehicle pursuits, and burglaries.
            </p>
            <p style='margin: 0 0 0.8rem 0; color: #1e40af;'>
                <strong>Why vectors?</strong> Converting text to vectors allows us to perform semantic search - finding reports
                by <em>meaning</em> rather than just keyword matching. Similar incidents will have vectors that are close together
                in the high-dimensional vector space.
            </p>
            <p style='margin: 0; color: #1e40af; font-size: 0.95rem;'>
                <strong>How it works:</strong> When you search, your query is also converted to a vector, and we calculate
                cosine similarity between your query vector and each report vector to find the most relevant matches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show example of text vs vector
    with st.expander("🔬 Example: Plain Text vs. Vector Representation", expanded=False):
        st.markdown(
            """
            **Original Text (Human-Readable):**
            ```
            "Vehicle pursuit terminated after suspect fled"
            ```

            **Vector Representation (Machine-Readable):**

            Below is what this sentence looks like when converted to a vector embedding. This is a simplified example
            showing just the first 20 dimensions (real embeddings typically have 1536 dimensions or more):

            ```python
            [
                -0.0234, 0.1456, -0.0891, 0.2103, -0.0567,
                 0.0923, -0.1234, 0.0456, -0.0789, 0.1567,
                -0.0345, 0.0678, -0.0912, 0.1234, -0.0456,
                 0.0789, -0.1023, 0.0567, -0.0234, 0.0891
                ... (continues for 1536+ dimensions)
            ]
            ```

            **Key Points:**
            - Each number captures a different aspect of the meaning
            - Similar words/phrases have similar patterns in their vectors
            - The model learned these representations from analyzing billions of text examples
            - Two sentences about vehicle pursuits will have vectors pointing in similar directions
            - Unrelated sentences (e.g., "shoplifting" vs "vehicle pursuit") will have vectors pointing in different directions

            **Similarity Calculation:**
            - Cosine similarity measures the angle between two vectors
            - Score of **1.0** = identical direction (same meaning)
            - Score of **0.0** = perpendicular (unrelated)
            - Score of **-1.0** = opposite direction (contradictory, rare in practice)
            """
        )

    # Database entries section
    st.markdown("### 📄 Database Records (Plain Text View)")
    st.markdown("_Below are the actual police reports in the vector collection - each has been converted to a vector for semantic search:_")

    # Build the display HTML for database entries
    import html

    # Add custom scrollbar styling
    st.markdown("""
        <style>
        .db-entries-container {
            background: #1e1e1e;
            padding: 0.8rem;
            border-radius: 8px;
            font-family: "Courier New", monospace;
            font-size: 0.85rem;
            color: #d4d4d4;
            max-height: 200px;
            overflow-y: scroll;
        }
        .db-entries-container::-webkit-scrollbar {
            width: 14px;
        }
        .db-entries-container::-webkit-scrollbar-track {
            background: #3d3d3d;
            border-radius: 6px;
        }
        .db-entries-container::-webkit-scrollbar-thumb {
            background: #d4d4d4;
            border-radius: 6px;
            border: 2px solid #3d3d3d;
        }
        .db-entries-container::-webkit-scrollbar-thumb:hover {
            background: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

    entries_html = "<div class='db-entries-container'>"

    for entry in DATABASE_ENTRIES:
        # Color coding for different report types
        color_map = {
            "Shoplifting": "#fbbf24",  # yellow
            "Assault": "#ef4444",  # red
            "Vehicle Pursuit": "#3b82f6",  # blue
            "Domestic Disturbance": "#f97316",  # orange
            "Burglary": "#a855f7",  # purple
            "Construction Theft": "#10b981",  # green
        }
        color = color_map.get(entry["title"], "#6b7280")

        # Escape the text to prevent HTML injection
        safe_text = html.escape(entry["text"])

        entries_html += f"<div style='margin-bottom: 0.4rem; padding: 0.3rem 0.5rem; background: #2d2d2d; border-radius: 3px; border-left: 3px solid {color};'><span style='color: #9ca3af;'>{safe_text}</span></div>"

    entries_html += "</div>"
    st.markdown(entries_html, unsafe_allow_html=True)

    st.markdown("---")

    # Row: Example buttons (4/12) and Search Query (8/12)
    col_examples, col_query = st.columns([4, 8])

    with col_examples:
        st.markdown("### 🔍 Try Example Searches")
        if st.button("🚗 Vehicle Pursuit", use_container_width=True):
            st.session_state["query_input"] = "Looking for incidents with a red truck fleeing the scene"
            st.rerun()
        if st.button("🏪 Theft Cases", use_container_width=True):
            st.session_state["query_input"] = "Reports involving stolen property or theft"
            st.rerun()
        if st.button("👊 Violent Incidents", use_container_width=True):
            st.session_state["query_input"] = "Physical violence or assault cases"
            st.rerun()

    with col_query:
        st.markdown("### 📌 Search Query")

        def on_query_change():
            """Auto-trigger search when text input changes (Enter key pressed)"""
            st.session_state["trigger_search"] = True

        base_text = st.text_input(
            "Base text for comparison",
            label_visibility="collapsed",
            help="Type your search query and press Enter to search",
            key="query_input",
            on_change=on_query_change,
            placeholder="Type your search query and press Enter...",
        )

        # Search button below query in same column
        if st.button("🔍 Search Database (Find Similar Reports)", use_container_width=True, type="primary"):
            st.session_state["trigger_search"] = True

    if "embedding_results" not in st.session_state:
        st.session_state["embedding_results"] = []

    # Perform search if triggered
    if st.session_state.get("trigger_search", False):
        st.session_state["trigger_search"] = False
        if not base_text.strip():
            st.warning("⚠️ Please provide a search query")
        else:
            # Defer the heavy embedding work to a spinner so the UI stays responsive.
            with st.spinner("🔄 Generating embeddings and calculating similarities..."):
                base_embedding = embed_text(base_text)
                results = []
                for entry in DATABASE_ENTRIES:
                    candidate_embedding = embed_text(entry["text"])
                    score = cosine_similarity(base_embedding, candidate_embedding)
                    results.append(
                        {
                            "title": entry["title"],
                            "text": entry["text"],
                            "score": round(score, 3),
                        }
                    )
                st.session_state["embedding_results"] = results

    st.markdown("---")

    if st.session_state["embedding_results"]:
        # Reuse the last computed embedding scores for display updates.
        results = st.session_state["embedding_results"]
        st.markdown("### 📊 Similarity Scores (All Database Entries)")

        # Create enhanced plotly chart using titles
        labels = [entry["title"] for entry in results]
        scores = [entry["score"] for entry in results]
        colors = [get_similarity_color(score) for score in scores]

        # Create custom hover text with full entry text
        hover_texts = [f"<b>{entry['title']}</b><br>Similarity: {entry['score']:.3f}<br><br>{entry['text'][:100]}..."
                      for entry in results]

        # Build a bar chart that shows cosine similarity per report type.
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=scores,
                text=[f"{score:.3f}" for score in scores],
                textposition="outside",
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ])

        fig.update_layout(
            yaxis=dict(
                range=[0, 1],
                title=dict(
                    text="Cosine Similarity Score",
                    font=dict(size=14),
                ),
                gridcolor='rgba(0,0,0,0.1)',
            ),
            xaxis=dict(
                title=dict(
                    text="Report Type",
                    font=dict(size=14),
                ),
                tickangle=0,
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            margin=dict(t=30, b=120, l=80, r=40),
            font=dict(size=13),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Results cards
        st.markdown("### 🏆 Top Matching Reports (Ranked by Similarity)")

        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        for idx, entry in enumerate(sorted_results):
            score = entry["score"]
            color = get_similarity_color(score)

            st.markdown(
                f"""
                <div style='background: white; padding: 1.2rem; border-radius: 10px;
                margin: 0.8rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-left: 4px solid {color};'>
                    <div style='display: flex; justify-content: space-between; align-items: start;'>
                        <div style='flex: 1;'>
                            <div style='color: #6b7280; font-size: 0.85rem; margin-bottom: 0.3rem;'>
                                Rank #{idx + 1}
                            </div>
                            <div style='color: #1f2937; font-size: 1rem; margin-bottom: 0.5rem;'>
                                "{entry["text"]}"
                            </div>
                            <div style='color: {color}; font-weight: 600;'>
                                {describe_similarity(score)}
                            </div>
                        </div>
                        <div style='background: {color}; color: white; padding: 0.8rem 1.2rem;
                        border-radius: 8px; font-size: 1.5rem; font-weight: 700; min-width: 80px;
                        text-align: center;'>
                            {score:.3f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Information expander
        with st.expander("ℹ️ How Vector Similarity Works"):
            st.markdown(
                """
                **What are Embeddings?**

                Embeddings convert text into high-dimensional vectors (lists of numbers) that represent the *meaning*
                of the text. Sentences with similar meanings have vectors that point in similar directions.

                **Cosine Similarity:**

                This measures how similar two vectors are on a scale from 0 to 1:
                - **1.0** = Identical or nearly identical meaning
                - **0.75-0.9** = Strong semantic similarity (related concepts)
                - **0.5-0.75** = Moderate similarity (some overlap)
                - **Below 0.5** = Little to no relationship

                **Why This Matters for Law Enforcement:**

                - Search by *concept* rather than exact keywords
                - "Red truck fleeing" will match "vehicle pursuit with pickup truck"
                - Different witnesses describe the same event differently - embeddings connect them
                - Quickly filter out unrelated incidents (assaults, burglaries) when searching for vehicle pursuits

                **Try Different Searches:**
                - "Theft of property" → Should match shoplifting and burglary
                - "Physical altercation" → Should match assault
                - "Vehicle escaping" → Should match the pursuit report
                """
            )
    else:
        st.info("💡 Click the search button above to find similar reports in the database using vector similarity")
