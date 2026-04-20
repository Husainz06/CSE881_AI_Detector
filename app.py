"""
AI Text Detector - Streamlit Application
CSE 881: Automated Classification of Human vs AI Postings
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import joblib
import plotly.express as px
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

ALL_MODELS = [
    "CatBoost",
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "XGBoost",
    "TinyBERT",
    "Custom Algorithm",
]

# Trained on the combined dataset (jobs + agricultural + social media)
RESULTS_DF = pd.DataFrame(
    {
        "Model": [
            "CatBoost",
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "XGBoost",
            "TinyBERT",
            "Custom Algorithm",
        ],
        "Accuracy": [0.9430, 0.9480, 0.9500, 0.9480, 0.9400, 0.9610, 0.6640],
        "Precision": [0.95, 0.95, 0.95, 0.95, 0.94, 0.96, 0.74],
        "Recall": [0.94, 0.95, 0.95, 0.95, 0.94, 0.96, 0.51],
        "F1": [0.94, 0.95, 0.95, 0.95, 0.94, 0.96, 0.60],
    }
)


# Brand-inspired colors for each text source
SOURCE_COLORS = {
    "human": "#64748B",  # slate (neutral)
    "claude": "#D97757",  # Anthropic coral
    "gemini": "#4285F4",  # Google blue
    "chatgpt": "#10A37F",  # OpenAI teal
    "copilot": "#8b5cf6",  # Microsoft Copilot violet
    "perplexity": "#20B8CD",  # Perplexity teal
    "openai/gpt-oss-120b": "#10A37F",  # OpenAI teal
    "qwen/qwen2.5-7b-instruct": "#F97316",  # Alibaba orange
    "mistralai/mixtral-8x22b-instruct-v0.1": "#FA520F",  # Mistral orange
    "meta/llama-3.1-70b-instruct": "#1877F2",  # Meta blue
}


@st.cache_data
def load_jobs_data():
    path = os.path.join(BASE_DIR, "scraping", "jobs", "combined_jobs.csv")
    return pd.read_csv(path)


@st.cache_data
def load_ag_data():
    human_path = os.path.join(
        BASE_DIR, "scraping", "agricultural", "human_listings.json"
    )
    ai_path = os.path.join(BASE_DIR, "scraping", "agricultural", "ai_listings.json")
    with open(human_path) as f:
        human = pd.DataFrame(json.load(f))
    human["label"] = "Human"
    human["source_model"] = "human"
    with open(ai_path) as f:
        ai = pd.DataFrame(json.load(f))
    ai["label"] = "AI"
    return pd.concat([human, ai], ignore_index=True)


@st.cache_data
def load_social_data():
    path = os.path.join(BASE_DIR, "scraping", "social_media", "Combined_Dataset.csv")
    return pd.read_csv(path)


# Text cleaning function (must match the notebook preprocessing)
def deep_clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords

        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    except Exception:
        return text


@st.cache_resource
def load_custom_algorithm_data():
    """Load the pre-trained AI word/bigram/trigram ratio dictionaries."""
    base = os.path.join(BASE_DIR, "custom algorithm")
    try:
        words_df = pd.read_csv(os.path.join(base, "ai_words.csv"))
        bigrams_df = pd.read_csv(os.path.join(base, "ai_bigrams.csv"))
        trigrams_df = pd.read_csv(os.path.join(base, "ai_trigrams.csv"))
    except FileNotFoundError:
        return None, None, None
    ai_words = dict(zip(words_df["word"], words_df["ratio"]))
    ai_bigrams = {
        tuple(k.split()): v for k, v in zip(bigrams_df["bigram"], bigrams_df["ratio"])
    }
    ai_trigrams = {
        tuple(k.split()): v
        for k, v in zip(trigrams_df["trigram"], trigrams_df["ratio"])
    }
    return ai_words, ai_bigrams, ai_trigrams


def _custom_detect(text, ai_words, ai_bigrams, ai_trigrams, threshold=0.2):
    """Custom AI detection: weighted word/bigram/trigram ratio scoring."""
    import nltk
    import string
    from nltk.corpus import stopwords

    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    stop = set(stopwords.words("english"))

    # Word score: remove stopwords + punctuation, look up each word in ai_words
    lowered = text.lower()
    tokens = nltk.word_tokenize(lowered)
    cleaned_words = [
        w.translate(str.maketrans("", "", string.punctuation))
        for w in tokens
        if w not in stop
    ]
    cleaned_words = [w for w in cleaned_words if w]
    word_score = (
        sum(ai_words.get(w, 0) for w in cleaned_words) / len(cleaned_words)
        if cleaned_words
        else 0.0
    )

    # N-gram scores: alphanumeric tokens, then n-grams
    alnum_tokens = [w for w in tokens if w.isalnum()]
    bigrams = list(nltk.ngrams(alnum_tokens, 2))
    trigrams = list(nltk.ngrams(alnum_tokens, 3))
    bigram_score = (
        sum(ai_bigrams.get(bg, 0) for bg in bigrams) / len(bigrams) if bigrams else 0.0
    )
    trigram_score = (
        sum(ai_trigrams.get(tg, 0) for tg in trigrams) / len(trigrams)
        if trigrams
        else 0.0
    )

    score = word_score * 0.5 + bigram_score * 0.3 + trigram_score * 0.2
    is_ai = score > threshold
    # Map raw score to a [0.5, 1.0] confidence via logistic centered on threshold
    z = (score - threshold) * 10
    prob_ai = 1.0 / (1.0 + np.exp(-z))
    confidence = float(prob_ai if is_ai else 1.0 - prob_ai)
    return bool(is_ai), confidence


@st.cache_resource
def load_model(model_name):
    """Load a trained model and its vectorizer from disk (combined dataset)."""
    sklearn_models = {
        "SVM": "svm_jobs.pkl",
        "Logistic Regression": "lr_jobs.pkl",
        "Random Forest": "rf_jobs.pkl",
    }

    if model_name == "CatBoost":
        path = os.path.join(MODELS_DIR, "catboost_jobs.cbm")
        if not os.path.exists(path):
            return None, None, None
        model = CatBoostClassifier()
        model.load_model(path)
        return model, None, "catboost"

    if model_name == "XGBoost":
        path = os.path.join(MODELS_DIR, "xgboost_jobs.json")
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_jobs.pkl")
        if not os.path.exists(path) or not os.path.exists(tfidf_path):
            return None, None, None
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None, None, None
        model = XGBClassifier()
        model.load_model(path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf, "tfidf"

    if model_name == "TinyBERT":
        path = os.path.join(MODELS_DIR, "tinybert_jobs.keras")
        if not os.path.exists(path):
            return None, None, None
        try:
            import keras
            import keras_nlp  # noqa: F401 (registers custom layers)

            model = keras.models.load_model(path)
        except Exception as e:
            return None, None, f"bert_error:{type(e).__name__}: {e}"
        return model, None, "bert"

    if model_name in sklearn_models:
        model_path = os.path.join(MODELS_DIR, sklearn_models[model_name])
        tfidf_path = os.path.join(MODELS_DIR, "tfidf_jobs.pkl")
        if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
            return None, None, None
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf, "tfidf"

    return None, None, None


def predict_text(text, model_name):
    """Run inference on user text and return (is_ai, confidence)."""
    if model_name == "Custom Algorithm":
        ai_words, ai_bigrams, ai_trigrams = load_custom_algorithm_data()
        if ai_words is None:
            return None, None
        return _custom_detect(text, ai_words, ai_bigrams, ai_trigrams)

    model, tfidf, model_type = load_model(model_name)
    if isinstance(model_type, str) and model_type.startswith("bert_error"):
        return None, model_type
    if model is None:
        return None, None

    cleaned = deep_clean_text(text)

    if model_type == "catboost":
        input_df = pd.DataFrame({"full_text": [cleaned]})
        pred = model.predict(input_df)
        proba = model.predict_proba(input_df)
        is_ai = (
            int(pred[0][0]) == 1
            if isinstance(pred[0], (list, np.ndarray))
            else int(pred[0]) == 1
        )
        confidence = float(max(proba[0]))
        return is_ai, confidence

    if model_type == "bert":
        # TinyBERT model takes raw light-cleaned text
        light_cleaned = re.sub(r"<.*?>", "", text)
        light_cleaned = re.sub(r"\s+", " ", light_cleaned).strip()
        logits = model.predict([light_cleaned], verbose=0)
        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
        is_ai = int(np.argmax(probs)) == 1
        confidence = float(max(probs))
        return is_ai, confidence

    if model_type == "tfidf":
        features = tfidf.transform([cleaned])
        pred = model.predict(features)
        is_ai = int(pred[0]) == 1
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            confidence = float(max(proba[0]))
        elif hasattr(model, "decision_function"):
            decision = float(model.decision_function(features)[0])
            confidence = 1.0 / (1.0 + np.exp(-abs(decision)))
        else:
            confidence = 1.0
        return is_ai, confidence

    return None, None


# Page Functions


def _sidebar_brand():
    """Consistent brand header across all sidebars."""
    st.markdown("### AI Text Detector")
    st.caption("CSE 881 · Spring 2025")
    st.divider()


def page_home():
    with st.sidebar:
        _sidebar_brand()
        st.markdown("**Quick Facts**")
        st.caption("4,996 labeled samples")
        st.caption("3 distinct domains")
        st.caption("6 trained models")
        st.caption("Best: TinyBERT at 96.1%")

    st.title("AI Text Detector")
    st.markdown(
        "A unified classifier trained on nearly 5,000 real and AI-generated samples "
        "across job postings, agricultural listings, and social media."
    )

    cta1, cta2, _ = st.columns([1, 1, 3])
    with cta1:
        if st.button(
            "Try the Detector",
            type="primary",
            icon=":material/document_scanner:",
            use_container_width=True,
        ):
            st.switch_page(PAGE_DETECTOR)
    with cta2:
        if st.button(
            "View Performance",
            icon=":material/leaderboard:",
            use_container_width=True,
        ):
            st.switch_page(PAGE_PERFORMANCE)

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Text Samples", "4,996")
    m2.metric("Datasets", "3")
    m3.metric("AI Sources", "9")
    m4.metric("Best Accuracy", "96.1%")

    st.divider()

    st.subheader("How It Works")
    st.caption("Four-stage pipeline from raw data to live predictions.")

    steps = [
        (
            "Collect",
            "Scrape real postings from Indeed, Care Farming Network, and Reddit.",
        ),
        (
            "Generate",
            "Synthesize AI text via Claude, ChatGPT, Gemini, Copilot, and more.",
        ),
        ("Preprocess", "Clean, normalize, and extract TF-IDF features with NLTK."),
        ("Classify", "Train 6 models and run real-time predictions in this app."),
    ]
    cols = st.columns(4, gap="large")
    for i, (col, (title, desc)) in enumerate(zip(cols, steps)):
        with col:
            st.caption(f"STEP {i + 1:02d}")
            st.markdown(f"##### {title}")
            st.caption(desc)

    st.divider()

    st.subheader("Datasets")
    st.caption("Three distinct domains, combined into a single training set.")

    datasets = [
        (
            "Job Postings",
            "3,000",
            "Indeed (human) + Claude, ChatGPT, Copilot, Gemini, Perplexity (AI)",
        ),
        (
            "Agricultural Listings",
            "790",
            "Care Farming Network (human) + 4 NVIDIA NIM models (AI)",
        ),
        (
            "Social Media Posts",
            "1,206",
            "Reddit (human) + ChatGPT, Claude, Gemini (AI)",
        ),
    ]
    cols = st.columns(3, gap="large")
    for col, (title, count, sources) in zip(cols, datasets):
        with col:
            st.markdown(f"##### {title}")
            st.metric("Samples", count)
            st.caption(sources)


MODEL_DESCRIPTIONS = {
    "CatBoost": "Gradient boosting on raw text. Fast inference, handles categorical features natively.",
    "Logistic Regression": "Linear baseline on TF-IDF features. Lightweight and interpretable.",
    "Random Forest": "Tree ensemble on TF-IDF. Robust to noise, captures non-linear patterns.",
    "SVM": "Support vector machine with linear kernel on TF-IDF.",
    "XGBoost": "Gradient boosting on TF-IDF. Strong general-purpose baseline.",
    "TinyBERT": "Fine-tuned transformer (bert_tiny_en_uncased). Highest accuracy at 96.1%.",
    "Custom Algorithm": "Heuristic scoring from AI-leaning word/bigram/trigram ratios (50/30/20 weighted).",
}


def page_detector():
    if "detector_text" not in st.session_state:
        st.session_state["detector_text"] = ""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar:
        _sidebar_brand()
        model_choice = st.selectbox("Model", ALL_MODELS)
        st.caption(MODEL_DESCRIPTIONS.get(model_choice, ""))

        st.divider()
        st.markdown("**Tips**")
        st.caption("Longer text yields more reliable predictions.")
        st.caption("Try different models to compare.")

        if st.session_state["history"]:
            st.divider()
            st.markdown("**Recent Predictions**")
            for entry in reversed(st.session_state["history"][-5:]):
                verdict = "AI" if entry["is_ai"] else "Human"
                st.caption(
                    f"**{verdict}** · {entry['confidence']:.0%} · {entry['model']}"
                )

    st.title("Live Detector")
    st.caption("Paste any text and classify it as human-written or AI-generated.")

    user_text = st.text_area(
        "Text to classify",
        height=260,
        placeholder="Paste any text here (job posting, agricultural listing, social media post, or anything else)...",
        label_visibility="collapsed",
        value=st.session_state["detector_text"],
        key="text_input_paste",
    )

    word_count = len(user_text.split()) if user_text else 0
    char_count = len(user_text) if user_text else 0
    st.caption(f"**{word_count:,}** words · **{char_count:,}** characters")

    classify_clicked = st.button(
        "Classify",
        icon=":material/document_scanner:",
        use_container_width=True,
        type="primary",
    )

    if classify_clicked:
        input_text = user_text or ""

        if not input_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner(f"Running {model_choice}..."):
                is_ai, confidence = predict_text(input_text, model_choice)

            if isinstance(confidence, str) and confidence.startswith("bert_error"):
                detail = confidence.split(":", 1)[1] if ":" in confidence else ""
                st.error(f"TinyBERT failed to load.\n\n**Error:** {detail}")
            elif is_ai is None:
                st.error(
                    "Model files not found. Run the notebook and execute the "
                    "model-saving cell to generate files in `models/`."
                )
            else:
                label = "AI-Generated" if is_ai else "Human-Written"

                with st.container(border=True):
                    r1, r2 = st.columns([2, 1])
                    with r1:
                        if is_ai:
                            st.error(f"### {label}")
                        else:
                            st.success(f"### {label}")
                        st.caption(f"Classified by **{model_choice}**")
                    with r2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    st.progress(float(confidence))

                st.session_state["history"].append(
                    {
                        "is_ai": is_ai,
                        "label": label,
                        "confidence": confidence,
                        "model": model_choice,
                    }
                )
                st.session_state["detector_text"] = ""


def page_performance():
    with st.sidebar:
        _sidebar_brand()
        st.markdown("**Filter Models**")
        selected_models = []
        for m in ALL_MODELS:
            if st.checkbox(m, value=True, key=f"perf_{m}"):
                selected_models.append(m)

        st.divider()
        st.markdown("**Test Set**")
        st.caption("500 human · 500 AI samples")
        st.caption("Held out from training data")

    filtered_df = RESULTS_DF[RESULTS_DF["Model"].isin(selected_models)]

    st.title("Model Performance")
    st.caption(
        "Evaluation on the combined dataset (jobs + agricultural + social media)."
    )

    st.divider()

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        return

    best_idx = filtered_df["Accuracy"].idxmax()
    best = filtered_df.loc[best_idx]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Top Model", best["Model"])
    b2.metric("Accuracy", f"{best['Accuracy']:.1%}")
    b3.metric("F1 Score", f"{best['F1']:.1%}")
    b4.metric("Precision", f"{best['Precision']:.1%}")

    st.divider()
    st.subheader("All Models")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(
        filtered_df.style.format({c: "{:.1%}" for c in metric_cols}).highlight_max(
            subset=metric_cols, color="#ef4444b3"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Visual Comparison")
    tab_bar, tab_cm = st.tabs(["Metrics", "Confusion Matrices"])

    with tab_bar:
        chart_df = filtered_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Score",
        )
        y_min = max(0.0, chart_df["Score"].min() - 0.05)
        y_low = 0.85 if y_min >= 0.80 else y_min
        fig = px.bar(
            chart_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "Accuracy": "#ef4444",
                "Precision": "#8b5cf6",
                "Recall": "#10b981",
                "F1": "#f59e0b",
            },
            range_y=[y_low, 1.0],
        )
        fig.update_layout(
            height=480,
            margin=dict(l=40, r=20, t=20, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ededed"),
            xaxis=dict(tickangle=0, title=None),
            yaxis=dict(
                gridcolor="#262626",
                zeroline=False,
                tickformat=".0%",
                title="Score",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                title=None,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_cm:
        # Confusion matrices on the combined test set (500 Human / 500 AI)
        all_cms = {
            "CatBoost": np.array([[492, 8], [49, 451]]),
            "Logistic Regression": np.array([[488, 12], [40, 460]]),
            "Random Forest": np.array([[493, 7], [43, 457]]),
            "SVM": np.array([[487, 13], [39, 461]]),
            "XGBoost": np.array([[489, 11], [49, 451]]),
            "TinyBERT": np.array([[492, 8], [31, 469]]),
            "Custom Algorithm": np.array([[411, 89], [247, 253]]),
        }
        labels = ["Human", "AI"]
        cms_to_show = {k: v for k, v in all_cms.items() if k in selected_models}

        cols = st.columns(min(len(cms_to_show), 3))
        for i, (name, cm) in enumerate(cms_to_show.items()):
            with cols[i % 3]:
                fig = px.imshow(
                    cm,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale=[
                        [0.0, "#171717"],
                        [0.3, "#171717"],
                        [0.6, "#fb923c"],
                        [1.0, "#ef4444"],
                    ],
                    aspect="equal",
                )
                fig.update_traces(
                    textfont=dict(size=18, color="white", family="Arial Black"),
                )
                fig.update_layout(
                    title=dict(
                        text=f"<b>{name}</b>",
                        font=dict(size=15, color="#ededed"),
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    margin=dict(l=50, r=20, t=50, b=50),
                    height=320,
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ededed", size=13),
                    xaxis=dict(side="bottom"),
                )
                st.plotly_chart(fig, use_container_width=True)


DATASET_CONFIG = {
    "Job Postings": {
        "loader": load_jobs_data,
        "label_col": "target",
        "source_col": "source_model",
        "text_col": "full_text",
        "label_map": {0: "Human", 1: "AI"},
        "description": "~3,000 data science job postings from Indeed plus AI-generated postings from 5 LLMs.",
        "schema": [
            ("target", "int", "0 = Human, 1 = AI"),
            ("source_model", "str", "Origin of the entry"),
            ("full_text", "str", "Title + description"),
        ],
    },
    "Agricultural Listings": {
        "loader": load_ag_data,
        "label_col": "label",
        "source_col": "source_model",
        "text_col": "description",
        "label_map": {"Human": "Human", "AI": "AI"},
        "description": "790 farm listings from the Care Farming Network plus AI-generated listings from 4 NVIDIA NIM models.",
        "schema": [
            ("id", "str", "URL-friendly slug"),
            ("name", "str", "Listing name"),
            ("description", "str", "Full description"),
            ("label", "str", "Human or AI"),
            ("source_model", "str", "Generation model"),
        ],
    },
    "Social Media": {
        "loader": load_social_data,
        "label_col": "target",
        "source_col": "source_model",
        "text_col": "Title",
        "label_map": {0: "Human", 1: "AI"},
        "description": "~1,200 Reddit posts from real subreddits plus AI-generated posts from ChatGPT, Claude, and Gemini.",
        "schema": [
            ("Title", "str", "Reddit post title"),
            ("URL", "str", "Source URL"),
            ("Score", "int", "Upvotes minus downvotes"),
            ("Upvote_Ratio", "float", "Ratio of upvotes"),
            ("Num_Comments", "int", "Number of comments"),
            ("Post_Date", "str", "Posting date"),
            ("target", "int", "0 = Human, 1 = AI"),
            ("source_model", "str", "Generation model"),
        ],
    },
}


def page_data():
    with st.sidebar:
        _sidebar_brand()
        st.markdown("**Dataset**")
        ds = st.selectbox(
            "Dataset", list(DATASET_CONFIG.keys()), label_visibility="collapsed"
        )
        st.caption(DATASET_CONFIG[ds]["description"])

        st.divider()
        st.markdown("**Filter Sample**")
        label_f = []
        if st.checkbox("Human", value=True, key="data_label_human"):
            label_f.append("Human")
        if st.checkbox("AI", value=True, key="data_label_ai"):
            label_f.append("AI")
        max_rows = st.slider("Max rows to display", 10, 500, 50)

    cfg = DATASET_CONFIG[ds]
    df = cfg["loader"]()

    # Normalize label column to Human/AI strings for consistent filtering
    df = df.copy()
    df["_label"] = df[cfg["label_col"]].map(cfg["label_map"])

    st.title("Dataset Explorer")
    st.caption(cfg["description"])

    st.divider()

    # Overview metrics
    total = len(df)
    n_human = (df["_label"] == "Human").sum()
    n_ai = (df["_label"] == "AI").sum()
    avg_words = df[cfg["text_col"]].fillna("").astype(str).str.split().str.len().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{total:,}")
    c2.metric("Human", f"{n_human:,}")
    c3.metric("AI", f"{n_ai:,}")
    c4.metric("Avg Words", f"{avg_words:.0f}")

    st.divider()

    # Side-by-side charts: sources + class balance
    col_sources, col_balance = st.columns([2, 1], gap="large")

    with col_sources:
        st.subheader("Distribution by Source")
        src_counts = df[cfg["source_col"]].value_counts().reset_index()
        src_counts.columns = ["Source", "Entries"]
        fig_src = px.bar(
            src_counts,
            x="Source",
            y="Entries",
            color="Source",
            color_discrete_map=SOURCE_COLORS,
            text="Entries",
        )
        fig_src.update_traces(textposition="outside", cliponaxis=False)
        fig_src.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ededed"),
            xaxis=dict(title=None, tickangle=-20),
            yaxis=dict(gridcolor="#262626", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig_src, use_container_width=True)

    with col_balance:
        st.subheader("Class Balance")
        balance_df = pd.DataFrame({"Label": ["Human", "AI"], "Count": [n_human, n_ai]})
        fig_bal = px.pie(
            balance_df,
            values="Count",
            names="Label",
            color="Label",
            color_discrete_map={"Human": "#10b981", "AI": "#ef4444"},
            hole=0.55,
        )
        fig_bal.update_traces(
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
        )
        fig_bal.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ededed"),
            showlegend=False,
        )
        st.plotly_chart(fig_bal, use_container_width=True)

    st.divider()

    # Text length distribution (Human vs AI)
    st.subheader("Text Length Distribution (Human vs AI)")
    word_counts = df[cfg["text_col"]].fillna("").astype(str).str.split().str.len()
    length_df = pd.DataFrame({"Words": word_counts, "Label": df["_label"]})
    fig_len = px.histogram(
        length_df,
        x="Words",
        color="Label",
        color_discrete_map={"Human": "#10b981", "AI": "#ef4444"},
        barmode="overlay",
        opacity=0.65,
        nbins=40,
    )
    fig_len.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ededed"),
        xaxis=dict(title="Word Count", gridcolor="#262626", zeroline=False),
        yaxis=dict(title="Frequency", gridcolor="#262626", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_len, use_container_width=True)

    st.divider()

    # Filtered sample table
    st.subheader("Sample Data")
    filtered = df[df["_label"].isin(label_f)].drop(columns=["_label"]).head(max_rows)
    st.caption(f"Showing {len(filtered):,} of {total:,} rows")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    with st.expander("Schema"):
        schema_df = pd.DataFrame(
            cfg["schema"], columns=["Field", "Type", "Description"]
        )
        st.dataframe(schema_df, use_container_width=True, hide_index=True)


# Configure Streamlit App

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGE_HOME = st.Page(
    page_home,
    title="Home",
    icon=":material/home:",
    url_path="home",
    default=True,
)
PAGE_DETECTOR = st.Page(
    page_detector,
    title="Detector",
    icon=":material/document_scanner:",
    url_path="detector",
)
PAGE_PERFORMANCE = st.Page(
    page_performance,
    title="Performance",
    icon=":material/leaderboard:",
    url_path="performance",
)
PAGE_DATA = st.Page(
    page_data, title="Data", icon=":material/database:", url_path="data"
)

nav = st.navigation(
    [PAGE_HOME, PAGE_DETECTOR, PAGE_PERFORMANCE, PAGE_DATA],
    position="top",
)

nav.run()
