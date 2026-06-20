"""
RoBERTa News NER — Streamlit Space app.

Loads Souvikbasur/roberta-base-news-ner from the Hugging Face Hub and
provides an interactive entity-tagging UI, an entity legend, a written
model report, and team credits.
"""

import html
import streamlit as st
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoBERTa News NER",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_ID = "Souvikbasur/roberta-base-news-ner"

# ──────────────────────────────────────────────────────────────────────────
# Entity schema — color palette
# Grouped by semantic family so colors are intuitive, not random:
#   people/orgs/groups -> warm (coral/amber/rose)
#   places/facilities  -> teal/cyan
#   time               -> violet
#   numbers/measures   -> blue/slate
#   legal/events/art   -> deep magenta/gold family
# Colors chosen for AA-readable text on both light and dark surfaces by
# pairing each fill with a near-black or near-white label as needed.
# ──────────────────────────────────────────────────────────────────────────
ENTITY_INFO = {
    "PER":   {"label": "Person",                         "emoji": "🧑",  "color": "#FF6F59"},
    "ORG":   {"label": "Organization",                    "emoji": "🏢",  "color": "#FFB000"},
    "NORP":  {"label": "Nationality / Religious / Political Group", "emoji": "🌐", "color": "#FF8FB1"},
    "LOC":   {"label": "Location",                         "emoji": "📍",  "color": "#14B8A6"},
    "FAC":   {"label": "Facility",                         "emoji": "🏛️", "color": "#06B6D4"},
    "DATE":  {"label": "Date",                             "emoji": "📅",  "color": "#A78BFA"},
    "TIME":  {"label": "Time",                             "emoji": "⏰",  "color": "#8B5CF6"},
    "EVT":   {"label": "Event",                            "emoji": "🎉",  "color": "#F472B6"},
    "PROD":  {"label": "Product",                          "emoji": "📦",  "color": "#FB923C"},
    "MONEY": {"label": "Monetary Value",                   "emoji": "💰",  "color": "#22C55E"},
    "PCT":   {"label": "Percentage",                       "emoji": "📊",  "color": "#84CC16"},
    "LAW":   {"label": "Law / Legal Reference",            "emoji": "⚖️", "color": "#C026D3"},
    "LANG":  {"label": "Language",                         "emoji": "🗣️", "color": "#EAB308"},
    "WOA":   {"label": "Work of Art",                      "emoji": "🎨",  "color": "#E879F9"},
    "QTY":   {"label": "Quantity",                         "emoji": "📏",  "color": "#38BDF8"},
    "ORD":   {"label": "Ordinal Number",                   "emoji": "🔢",  "color": "#60A5FA"},
    "CARD":  {"label": "Cardinal Number",                  "emoji": "#️⃣", "color": "#3B82F6"},
}

DEFAULT_COLOR = "#9CA3AF"

# Per-tag held-out test F1, from the model card (FacebookAI/roberta-base run)
TAG_F1 = {
    "PER": 0.9501, "LOC": 0.9425, "NORP": 0.9273, "PCT": 0.9055,
    "MONEY": 0.8869, "ORD": 0.8816, "CARD": 0.8791, "DATE": 0.8627,
    "ORG": 0.8404, "PROD": 0.8094, "TIME": 0.7831, "LANG": 0.7719,
    "QTY": 0.7005, "LAW": 0.7417, "EVT": 0.7407, "FAC": 0.6316, "WOA": 0.5732,
}

EXAMPLE_SENTENCES = [
    "Narendra Modi met officials in New Delhi on Monday regarding the new defence policy.",
    "The United Nations Security Council discussed the proposed law on climate change in Geneva.",
    "Apple announced a 12% increase in quarterly revenue, reaching $94 billion this year.",
    "The European Union signed a trade agreement with Japan during the G20 summit in Osaka.",
]

# ──────────────────────────────────────────────────────────────────────────
# Model loading — cached so it only loads once per Space instance
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return pipeline(
        "token-classification",
        model=MODEL_ID,
        aggregation_strategy="simple",
    )


# ──────────────────────────────────────────────────────────────────────────
# Styling — injected CSS. Streamlit's native theme (light/dark, switchable
# via the menu in the top-right or config.toml) drives background/text
# colors; this CSS layers tag chips, cards, and type treatment on top in a
# way that holds up under both.
# ──────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.ner-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 0.2rem;
}

.ner-subhead {
    font-size: 1.05rem;
    opacity: 0.75;
    margin-bottom: 1.6rem;
    font-weight: 500;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.3rem;
}

.wire-output {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.08rem;
    line-height: 2.1;
    padding: 1.6rem;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.045);
}

.tag-chip {
    display: inline-block;
    padding: 0.08em 0.5em;
    border-radius: 5px;
    font-weight: 700;
    color: #0B0D12;
    white-space: nowrap;
    margin: 0 1px;
}

.tag-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62em;
    font-weight: 700;
    vertical-align: super;
    margin-left: 0.2em;
    letter-spacing: 0.03em;
}

.legend-card {
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.22);
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    flex-shrink: 0;
}

.team-card {
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.22);
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.7rem;
}

.team-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.65;
}

.metric-box {
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.22);
    padding: 0.9rem 1rem;
    text-align: center;
}

.metric-box .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
}

.metric-box .label {
    font-size: 0.78rem;
    opacity: 0.65;
    margin-top: 0.15rem;
}

hr.thin {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.2);
    margin: 1.4rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_tagged_text(entities, original_text):
    """Build HTML with bright color-coded entity chips inline in the text."""
    if not entities:
        return html.escape(original_text)

    entities = sorted(entities, key=lambda e: e["start"])
    out = []
    cursor = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        if start < cursor:
            continue  # skip overlaps defensively
        out.append(html.escape(original_text[cursor:start]))
        tag = ent["entity_group"]
        info = ENTITY_INFO.get(tag, {"color": DEFAULT_COLOR, "emoji": "🏷️"})
        word = html.escape(original_text[start:end])
        out.append(
            f'<span class="tag-chip" style="background:{info["color"]}">'
            f'{word}<span class="tag-label">{info["emoji"]} {tag}</span>'
            f'</span>'
        )
        cursor = end
    out.append(html.escape(original_text[cursor:]))
    return "".join(out)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗞️ RoBERTa News NER")
    st.markdown(
        "An entity-tagging model fine-tuned on political, business, "
        "defence, crime, and international news text."
    )
    st.markdown("---")
    st.markdown("**Model on Hugging Face**")
    st.markdown(f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID})")
    st.markdown("---")
    st.markdown("**Quick facts**")
    st.markdown(
        "- Backbone: `FacebookAI/roberta-base`\n"
        "- 17 entity types\n"
        "- Held-out test micro F1: **0.8788**\n"
        "- Params: ~125M"
    )
    st.markdown("---")
    st.caption("💡 Tip: switch light/dark theme from the menu (⋮) top-right.")


# ──────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────
tab_analyze, tab_legend, tab_report, tab_team = st.tabs(
    ["🔍 Analyze", "🎨 Entity Legend", "📊 Model Report", "👥 Team"]
)

# ── TAB 1 — Analyze ─────────────────────────────────────────────────────
with tab_analyze:
    st.markdown('<div class="ner-headline">Tag entities in your text</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ner-subhead">Paste a news sentence below — people, places, '
        'organizations, dates, and 13 more entity types light up instantly.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1, 1])
    chosen_example = None
    for i, ex in enumerate(EXAMPLE_SENTENCES):
        with cols[i]:
            if st.button(f"📰 Example {i + 1}", use_container_width=True, key=f"ex_{i}"):
                chosen_example = ex

    default_text = chosen_example or st.session_state.get("ner_text", EXAMPLE_SENTENCES[0])

    text_input = st.text_area(
        "Your sentence",
        value=default_text,
        height=120,
        key="ner_text",
        label_visibility="collapsed",
        placeholder="Type or paste a news sentence here…",
    )

    run = st.button("✨ Tag entities", type="primary")

    if run and text_input.strip():
        with st.spinner("Loading model…" if "ner_pipe_loaded" not in st.session_state else "Tagging…"):
            ner = load_pipeline()
            st.session_state["ner_pipe_loaded"] = True
            results = ner(text_input)

        st.markdown('<div class="eyebrow">Tagged output</div>', unsafe_allow_html=True)
        tagged_html = render_tagged_text(results, text_input)
        st.markdown(f'<div class="wire-output">{tagged_html}</div>', unsafe_allow_html=True)

        if results:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Detected entities</div>', unsafe_allow_html=True)

            n_cols = 3
            ent_cols = st.columns(n_cols)
            for idx, ent in enumerate(results):
                tag = ent["entity_group"]
                info = ENTITY_INFO.get(tag, {"label": tag, "emoji": "🏷️", "color": DEFAULT_COLOR})
                with ent_cols[idx % n_cols]:
                    st.markdown(
                        f"""
                        <div class="legend-card">
                            <div class="legend-swatch" style="background:{info['color']}"></div>
                            <div>
                                <b>{info['emoji']} {html.escape(ent['word'])}</b><br>
                                <span style="opacity:0.7; font-size:0.85em;">
                                    {info.get('label', tag)} · {ent['score']:.0%} confidence
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No entities detected in this text.")
    elif run:
        st.warning("Please enter some text first.")

# ── TAB 2 — Entity Legend ───────────────────────────────────────────────
with tab_legend:
    st.markdown('<div class="ner-headline">The 17-tag schema</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ner-subhead">Every entity type the model recognizes, '
        'grouped by family, with its color used in the tagger above.</div>',
        unsafe_allow_html=True,
    )

    families = [
        ("People, organizations & groups", ["PER", "ORG", "NORP"]),
        ("Places", ["LOC", "FAC"]),
        ("Time", ["DATE", "TIME"]),
        ("Numbers & measures", ["MONEY", "PCT", "QTY", "ORD", "CARD"]),
        ("Legal, events & culture", ["LAW", "EVT", "WOA", "PROD", "LANG"]),
    ]

    for family_name, tags in families:
        st.markdown(f"**{family_name}**")
        cols = st.columns(min(len(tags), 4))
        for i, tag in enumerate(tags):
            info = ENTITY_INFO[tag]
            with cols[i % len(cols)]:
                st.markdown(
                    f"""
                    <div class="legend-card">
                        <div class="legend-swatch" style="background:{info['color']}"></div>
                        <div>
                            <b>{info['emoji']} {tag}</b><br>
                            <span style="opacity:0.7; font-size:0.85em;">{info['label']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

# ── TAB 3 — Model Report ────────────────────────────────────────────────
with tab_report:
    st.markdown('<div class="ner-headline">Model report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ner-subhead">FacebookAI/roberta-base, fine-tuned for political/news NER.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("87.88%", "Micro F1"),
        ("86.41%", "Precision"),
        ("89.41%", "Recall"),
        ("81.34%", "Macro F1"),
    ]
    for col, (val, lab) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="value">{val}</div>'
                f'<div class="label">{lab}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    st.markdown("#### Per-entity performance (held-out test F1)")
    st.markdown(
        "Sorted highest to lowest. Color bars use the same palette as the tagger."
    )

    sorted_tags = sorted(TAG_F1.items(), key=lambda x: x[1], reverse=True)
    for tag, f1 in sorted_tags:
        info = ENTITY_INFO.get(tag, {"color": DEFAULT_COLOR, "emoji": "🏷️", "label": tag})
        bar_pct = int(f1 * 100)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.45rem;">
                <div style="width:120px; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                    {info['emoji']} {tag}
                </div>
                <div style="flex:1; background:rgba(128,128,128,0.15); border-radius:5px; height:14px; overflow:hidden;">
                    <div style="width:{bar_pct}%; background:{info['color']}; height:100%;"></div>
                </div>
                <div style="width:54px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                    {f1:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    st.markdown("#### Training configuration")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.markdown(
            """
            | Hyperparameter | Value |
            |---|---|
            | Base model | `FacebookAI/roberta-base` |
            | Max sequence length | 256 |
            | Learning rate | 2e-5 |
            | Effective batch size | 32 (2 × 16 grad. accum.) |
            | Epochs | 8 (early stopping, patience 2) |
            """
        )
    with config_col2:
        st.markdown(
            """
            | Hyperparameter | Value |
            |---|---|
            | Warmup ratio | 0.06 |
            | Weight decay | 0.01 |
            | Loss | Weighted CrossEntropy (3× rare tags) |
            | Optimizer | AdamW |
            | Seed | 42 |
            """
        )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    st.markdown("#### Interpretation")
    st.markdown(
        """
        Performance is strongest on high-frequency, well-bounded categories —
        **PER**, **LOC**, **NORP**, **PCT**, and **MONEY** — all scoring near or
        above 0.90 F1. A weighted loss strategy (3× weight on rare tags) helped
        lift historically difficult classes like **LAW**, **EVT**, and **FAC**.

        **WOA** (Work of Art) remains the most challenging category. Analysis
        across multiple backbone comparisons on this dataset attributes this
        less to raw data sparsity and more to **annotation noise and semantic
        boundary overlap** with adjacent categories — titles, named works, and
        policy/product references are sometimes inconsistently distinguished
        in the source annotations.

        The test F1 (0.8788) slightly exceeds the validation F1 (0.8739),
        indicating the model generalizes well with no signs of overfitting.
        """
    )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    st.markdown("#### Backbone comparison")
    st.markdown(
        "RoBERTa was selected as the best-performing backbone among four "
        "transformer architectures evaluated on the same merged 17-tag dataset:"
    )
    st.markdown(
        """
        | Backbone | Micro F1 |
        |---|---:|
        | **RoBERTa-base** (this model) | **0.8788** |
        | ELECTRA-base-discriminator | 0.8718 |
        | DeBERTa-v3-base | 0.8661 |
        | BERT-base (dslim/bert-base-NER) | 0.8617 |
        """
    )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    st.markdown("#### Intended use & limitations")
    st.markdown(
        """
        **Intended for:** extracting structured entities from English-language
        political, business, defence, crime, and international news text — for
        automated news monitoring, media analysis, or knowledge-base pipelines.

        **Not intended for:** non-English text, domains substantially different
        from news (legal contracts, medical records, social media slang), or
        high-stakes automated decisions without human review.
        """
    )

# ── TAB 4 — Team ────────────────────────────────────────────────────────
with tab_team:
    st.markdown('<div class="ner-headline">Team</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ner-subhead">The people behind this NER project.</div>',
        unsafe_allow_html=True,
    )

    TEAM = [
        ("Souvik Basu Roy", "Project Lead", "🧭"),
        ("Ishana Chatterjee", "Associate Lead", "🧩"),
        ("Debjani Paul", "Associate Lead", "🧩"),
        ("Amishi Shekhar", "Contributor", "⚙️"),
        ("Shreyashi Ghosh", "Contributor", "⚙️"),
    ]

    cols = st.columns(2)
    for i, (name, role, emoji) in enumerate(TEAM):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="team-card">
                    <div style="font-size:1.5rem;">{emoji}</div>
                    <div style="font-weight:700; font-size:1.05rem; margin-top:0.2rem;">{name}</div>
                    <div class="team-role">{role}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.markdown(
        "This NER system covers political, business/economic, defence, crime, "
        "and international news — one domain group (**G1**) within a larger "
        "multi-team project using a shared 17-tag entity schema across "
        "additional domains (sports, entertainment, technology, science, and more)."
    )

# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='thin'>", unsafe_allow_html=True)
st.caption(
    f"🤗 Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · "
    "Built with Streamlit · Runs on CPU"
)
