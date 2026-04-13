"""
AI Text Detector - Streamlit Application
CSE 881: Automated Classification of Human vs AI Postings
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_MODELS = [
    "CatBoost",
    "Random Forest",
    "Logistic Regression",
    "SVM",
    "LSTM",
    "BERT",
]

RESULTS_JOBS = pd.DataFrame(
    {
        "Model": ["CatBoost", "SVM", "Logistic Regression", "Random Forest", "BERT"],
        "Accuracy": [1.0000, 1.0000, 0.9987, 1.0000, 0.9947],
        "Precision": [1.00, 1.00, 1.00, 1.00, 0.99],
        "Recall": [1.00, 1.00, 1.00, 1.00, 0.99],
        "F1": [1.00, 1.00, 1.00, 1.00, 0.99],
    }
)

RESULTS_AG = pd.DataFrame(
    {
        "Model": ["SVM", "LSTM", "BERT"],
        "Accuracy": [0.9937, 0.8861, 1.0000],
        "Precision": [0.99, 0.90, 1.00],
        "Recall": [0.99, 0.89, 1.00],
        "F1": [0.99, 0.89, 1.00],
    }
)

# Combined view (average across datasets where a model was evaluated on both)
RESULTS_DF = pd.DataFrame(
    {
        "Model": ["CatBoost", "Random Forest", "Logistic Regression", "SVM", "LSTM", "BERT"],
        "Accuracy": [1.0000, 1.0000, 0.9987, 0.9969, 0.8861, 0.9974],
        "Precision": [1.00, 1.00, 1.00, 1.00, 0.90, 1.00],
        "Recall": [1.00, 1.00, 1.00, 1.00, 0.89, 1.00],
        "F1": [1.00, 1.00, 1.00, 1.00, 0.89, 0.99],
    }
)


@st.cache_data
def load_jobs_data():
    path = os.path.join(BASE_DIR, "scraping", "jobs", "combined.csv")
    return pd.read_csv(path)


@st.cache_data
def load_ag_data():
    human_path = os.path.join(BASE_DIR, "scraping", "agricultural", "human_listings.json")
    ai_path = os.path.join(BASE_DIR, "scraping", "agricultural", "ai_listings.json")
    with open(human_path) as f:
        human = pd.DataFrame(json.load(f))
    human["label"] = "Human"
    human["source_model"] = "human"
    with open(ai_path) as f:
        ai = pd.DataFrame(json.load(f))
    ai["label"] = "AI"
    return pd.concat([human, ai], ignore_index=True)


import re
import joblib
from catboost import CatBoostClassifier

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Text cleaning function (must match the notebook preprocessing)
def deep_clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
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
def load_model(domain, model_name):
    """Load a trained model and its vectorizer from disk."""
    suffix = "jobs" if domain == "Job Postings" else "ag"
    model_map = {
        "SVM": f"svm_{suffix}.pkl",
        "Logistic Regression": f"lr_{suffix}.pkl",
        "Random Forest": f"rf_{suffix}.pkl",
    }

    if model_name == "CatBoost":
        model_path = os.path.join(MODELS_DIR, f"catboost_{suffix}.cbm")
        if not os.path.exists(model_path):
            return None, None, "catboost"
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model, None, "catboost"

    if model_name in model_map:
        model_path = os.path.join(MODELS_DIR, model_map[model_name])
        tfidf_path = os.path.join(MODELS_DIR, f"tfidf_{suffix}.pkl")
        if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
            return None, None, None
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf, "tfidf"

    return None, None, None


def predict_text(text, domain, model_name):
    """Run inference on user text and return (is_ai, confidence)."""
    model, tfidf, model_type = load_model(domain, model_name)
    if model is None:
        return None, None

    cleaned = deep_clean_text(text)

    if model_type == "catboost":
        input_df = pd.DataFrame({"full_text": [cleaned]})
        pred = model.predict(input_df)
        proba = model.predict_proba(input_df)
        is_ai = int(pred[0][0]) == 1 if isinstance(pred[0], (list, np.ndarray)) else int(pred[0]) == 1
        confidence = float(max(proba[0]))
        return is_ai, confidence

    if model_type == "tfidf":
        features = tfidf.transform([cleaned])
        pred = model.predict(features)
        is_ai = int(pred[0]) == 1
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)
            confidence = float(max(proba[0]))
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(features)
            confidence = min(abs(float(decision[0])) / 2.0, 1.0)
        else:
            confidence = 1.0
        return is_ai, confidence

    return None, None

EXAMPLE_TEXTS = {
    "Job Postings": {
        "Human": (
            "Overview: About The Geneva Foundation\n"
            "The Geneva Foundation is a 501(c)(3) nonprofit established in 1993 "
            "with the mission to advance military medicine. We work alongside "
            "clinicians and scientists at military medical centers and clinics "
            "around the world. Our research covers a wide range of areas, from "
            "traumatic brain injury and infectious diseases to psychological health."
        ),
        "AI": (
            "We are a growing, innovation-focused company that leverages data science, "
            "machine learning, and modern software engineering practices to build trusted "
            "products and services used by millions of end users. Our organization is a "
            "well-established technology and analytics company that supports a wide range "
            "of commercial and public-sector clients. We emphasize rigorous data-driven "
            "decision making and invest heavily in scalable analytics platforms."
        ),
    },
    "Agricultural Listings": {
        "Human": (
            "We are a small family farm in rural Vermont, raising heritage breed "
            "chickens and growing mixed vegetables on 12 acres. My husband and I "
            "started this place in 2008 after he came back from his second tour. "
            "The farm has been good therapy for both of us. We sell at the Saturday "
            "market in Burlington and run a small CSA program."
        ),
        "AI": (
            "Nestled in the heart of California's rural landscape, the Reentry Farmer "
            "Project is a small, family-run homestead dedicated to therapeutic agriculture "
            "and social farming. Our mission is to support formerly incarcerated individuals "
            "as they reenter society through the healing power of working the land. We focus "
            "on urban rooftop gardening and hydroponics, providing not just fresh produce but "
            "also a sense of purpose and community."
        ),
    },
}


# Page Functions


def page_home():
    with st.sidebar:
        st.subheader("Project Info")
        st.caption("CSE 881 — Spring 2025")
        st.markdown(
            "Automated classification of human vs. AI-generated "
            "text across two domains: job postings and agricultural listings."
        )
        st.divider()
        st.markdown("**Datasets**")
        st.markdown("- Job Postings: ~2,000")
        st.markdown("- Agricultural: 790")
        st.markdown("**AI Sources:** 9 models")

    st.header("AI Text Detector")
    st.caption(
        "Classify human vs. AI-generated text across job postings and agricultural listings."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Text Samples", "2,800+")
    m2.metric("Datasets", "2")
    m3.metric("AI Sources", "9")
    m4.metric("Models Trained", "6+")

    st.divider()

    st.subheader("Pipeline")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("**1. Collect**")
        st.caption("Scrape real postings from Indeed & Care Farming Network.")
    with s2:
        st.markdown("**2. Generate**")
        st.caption("Produce AI text via Claude, ChatGPT, Gemini, Copilot & more.")
    with s3:
        st.markdown("**3. Preprocess**")
        st.caption("Clean, normalize, extract features with NLTK & TF-IDF.")
    with s4:
        st.markdown("**4. Classify**")
        st.caption("Train models and detect AI text in real time.")

    st.divider()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Job Postings Dataset")
        st.markdown("""
        ~2,000 data science job postings. Human entries scraped from Indeed;
        AI entries generated by **5 LLMs** (Claude, ChatGPT, Copilot, Gemini, Perplexity).

        Fields: title, location, salary, description, label, source.
        """)

    with col_right:
        st.subheader("Agricultural Listings Dataset")
        st.markdown("""
        790 farm & agricultural listings. Human entries scraped from the
        Care Farming Network; AI entries generated by **4 NVIDIA NIM models**.

        Fields: id, name, description, label, source model.
        """)


def page_detector():
    if "detector_text" not in st.session_state:
        st.session_state["detector_text"] = ""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar:
        st.subheader("Settings")
        domain = st.selectbox("Domain", ["Job Postings", "Agricultural Listings"])
        available_models = {
            "Job Postings": ["CatBoost", "SVM", "Logistic Regression", "Random Forest"],
            "Agricultural Listings": ["SVM"],
        }
        model_choice = st.selectbox("Model", available_models.get(domain, ALL_MODELS))

        st.divider()

        st.subheader("Try an Example")
        st.caption("Load sample text into the editor.")
        examples = EXAMPLE_TEXTS[domain]
        if st.button("Load Human example", use_container_width=True):
            st.session_state["detector_text"] = examples["Human"]
            st.rerun()
        if st.button("Load AI example", use_container_width=True):
            st.session_state["detector_text"] = examples["AI"]
            st.rerun()

        if st.session_state["history"]:
            st.divider()
            st.subheader("History")
            for entry in reversed(st.session_state["history"][-5:]):
                verdict = "AI" if entry["is_ai"] else "Human"
                st.caption(f"{verdict} ({entry['confidence']:.0%}) — {entry['model']}")

    st.header("Live Detector")
    st.caption("Paste text below and classify it as human-written or AI-generated.")

    user_text = ""
    job_summary = ""

    if domain == "Job Postings":
        tab_text, tab_fields = st.tabs(["Paste text", "Structured fields"])

        with tab_text:
            user_text = st.text_area(
                "Full posting text",
                height=220,
                placeholder="Paste the job posting here...",
                label_visibility="collapsed",
                value=st.session_state["detector_text"],
                key="text_input_paste",
            )

        with tab_fields:
            f1, f2 = st.columns(2)
            with f1:
                job_title = st.text_input("Title", placeholder="Senior Data Scientist")
                job_location = st.text_input("Location", placeholder="New York, NY")
            with f2:
                job_salary = st.text_input(
                    "Salary", placeholder="$120,000 - $150,000/yr"
                )
            job_summary = st.text_area(
                "Description", height=180, placeholder="Full job description..."
            )
    else:
        user_text = st.text_area(
            "Listing description",
            height=220,
            placeholder="Paste the agricultural listing here...",
            label_visibility="collapsed",
            value=st.session_state["detector_text"],
            key="text_input_ag",
        )

    classify_clicked = st.button("Classify", use_container_width=True)

    if classify_clicked:
        # Determine the text to classify
        input_text = user_text if user_text else ""
        if domain == "Job Postings" and not input_text and job_summary:
            input_text = job_summary

        if not input_text or not input_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.status("Classifying text...", expanded=True) as status:
                st.write("Preprocessing text...")
                st.write(f"Running **{model_choice}** model...")
                is_ai, confidence = predict_text(input_text, domain, model_choice)
                status.update(label="Classification complete", state="complete")

            if is_ai is None:
                st.error(
                    f"Model files not found. Run the notebook and execute the "
                    f"model-saving cell to generate files in `models/`."
                )
            else:
                label = "AI-Generated" if is_ai else "Human-Written"
                st.toast(f"Result: {label} ({confidence:.0%})", icon=":material/check_circle:")

                if is_ai:
                    st.error(f"**{label}** — {confidence:.0%} confidence ({model_choice})")
                else:
                    st.success(f"**{label}** — {confidence:.0%} confidence ({model_choice})")

                st.badge(
                    label,
                    icon=":material/smart_toy:" if is_ai else ":material/person:",
                    color="red" if is_ai else "green",
                )

                st.session_state["history"].append(
                    {
                        "is_ai": is_ai,
                        "label": label,
                        "confidence": confidence,
                        "model": model_choice,
                    }
                )
                st.session_state["detector_text"] = ""

                st.divider()
                st.caption("Was this classification accurate?")
                st.feedback("thumbs", key="detector_feedback")


def page_performance():
    with st.sidebar:
        st.subheader("Filters")
        perf_dataset = st.selectbox(
            "Dataset", ["Job Postings", "Agricultural Listings", "Combined"]
        )
        st.divider()
        st.subheader("Models")
        selected_models = []
        for m in ALL_MODELS:
            if st.checkbox(m, value=True, key=f"perf_{m}"):
                selected_models.append(m)

    if perf_dataset == "Job Postings":
        base_df = RESULTS_JOBS
    elif perf_dataset == "Agricultural Listings":
        base_df = RESULTS_AG
    else:
        base_df = RESULTS_DF

    filtered_df = base_df[base_df["Model"].isin(selected_models)]

    st.header("Model Performance")
    st.caption(f"Evaluation on **{perf_dataset}** dataset.")

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
        return

    best_idx = filtered_df["Accuracy"].idxmax()
    best = filtered_df.loc[best_idx]

    b1, b2, b3 = st.columns(3)
    b1.metric("Best Model", best["Model"])
    b2.metric("Accuracy", f"{best['Accuracy']:.1%}")
    b3.metric("F1 Score", f"{best['F1']:.3f}")

    st.divider()

    st.subheader("All Models")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    st.dataframe(
        filtered_df.style.format({c: "{:.1%}" for c in metric_cols}).highlight_max(
            subset=metric_cols, color="#ff4b4b40"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Visual Comparison")
    tab_bar, tab_cm = st.tabs(["Metrics", "Confusion Matrices"])

    with tab_bar:
        chart_df = filtered_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1"]
        ]
        st.bar_chart(
            chart_df, height=380, color=["#ff4b4b", "#ff6b6b", "#ff8a8a", "#ffaaaa"]
        )

    with tab_cm:
        # Confusion matrices from trained models
        cms_jobs = {
            "CatBoost": np.array([[378, 0], [0, 380]]),
            "Random Forest": np.array([[378, 0], [0, 380]]),
            "Logistic Regression": np.array([[377, 1], [0, 380]]),
            "SVM": np.array([[378, 0], [0, 380]]),
            "BERT": np.array([[374, 4], [0, 380]]),
        }
        cms_ag = {
            "SVM": np.array([[77, 1], [0, 80]]),
            "LSTM": np.array([[38, 1], [8, 32]]),
            "BERT": np.array([[39, 0], [0, 40]]),
        }
        if perf_dataset == "Job Postings":
            all_cms = cms_jobs
        elif perf_dataset == "Agricultural Listings":
            all_cms = cms_ag
        else:
            all_cms = {**cms_jobs, **cms_ag}
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
                    color_continuous_scale=[[0, "#262730"], [1, "#ff4b4b"]],
                    aspect="equal",
                )
                fig.update_layout(
                    title=dict(text=name, font=dict(size=14)),
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    margin=dict(l=40, r=20, t=40, b=40),
                    height=280,
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#fafafa"),
                )
                st.plotly_chart(fig, use_container_width=True)



def page_data():
    with st.sidebar:
        st.subheader("Filters")
        ds = st.selectbox("Dataset", ["Job Postings", "Agricultural Listings"])
        st.divider()

        if ds == "Job Postings":
            label_f = st.multiselect("Label", ["Human", "AI"], default=["Human", "AI"])
            source_f = st.multiselect(
                "Source",
                [
                    "human",
                    "chatgpt ai",
                    "claude ai",
                    "copilot ai",
                    "gemini ai",
                    "perplexity ai",
                ],
                default=[
                    "human",
                    "chatgpt ai",
                    "claude ai",
                    "copilot ai",
                    "gemini ai",
                    "perplexity ai",
                ],
            )
            max_rows = st.slider("Max rows", 10, 500, 100)
        else:
            label_f = st.multiselect(
                "Label", ["Human", "AI"], default=["Human", "AI"], key="ag_label"
            )
            model_f = st.multiselect(
                "AI Model",
                [
                    "openai/gpt-oss-120b",
                    "qwen/qwen2.5-7b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "mistralai/mixtral-8x22b-instruct-v0.1",
                ],
                default=[
                    "openai/gpt-oss-120b",
                    "qwen/qwen2.5-7b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "mistralai/mixtral-8x22b-instruct-v0.1",
                ],
            )
            max_rows = st.slider("Max rows", 10, 500, 100, key="ag_rows")

    st.header("Dataset Explorer")
    st.caption("Browse, filter, and inspect training data.")

    st.divider()

    if ds == "Job Postings":
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", "~2,000")
        c2.metric("Human", "~1,000")
        c3.metric("AI", "~1,000")

        col_table, col_chart = st.columns([3, 2], gap="large")

        with col_table:
            st.subheader("Sources")
            src_df = pd.DataFrame(
                {
                    "Source": [
                        "Human (Indeed)",
                        "ChatGPT",
                        "Claude",
                        "Copilot",
                        "Gemini",
                        "Perplexity",
                    ],
                    "Count": ["~1,000", "~200", "~200", "~200", "~200", "~200"],
                }
            )
            st.dataframe(src_df, use_container_width=True, hide_index=True)

        with col_chart:
            st.subheader("Distribution")
            chart_src = pd.DataFrame(
                {
                    "Source": [
                        "Human",
                        "ChatGPT",
                        "Claude",
                        "Copilot",
                        "Gemini",
                        "Perplexity",
                    ],
                    "Entries": [1000, 200, 200, 200, 200, 200],
                }
            ).set_index("Source")
            st.bar_chart(chart_src, height=250, color="#ff4b4b")

        st.divider()

        st.subheader("Sample Data")
        st.caption(
            f"Showing: labels={label_f}, sources={source_f}, limit={max_rows}"
        )
        jobs_df = load_jobs_data()
        label_map = {"Human": 0, "AI": 1}
        label_vals = [label_map[lbl] for lbl in label_f]
        filtered_jobs = jobs_df[
            (jobs_df["is_AI"].isin(label_vals)) & (jobs_df["source"].isin(source_f))
        ].head(max_rows)
        st.dataframe(filtered_jobs, use_container_width=True, hide_index=True)

        with st.expander("Schema"):
            st.markdown("""
| Field | Type | Description |
|---|---|---|
| `job_title` | str | Title of the posting |
| `job_location` | str | City, State (normalized) |
| `job_salary` | str | Annual salary range |
| `job_summary` | str | Full description |
| `is_AI` | int | 0 = Human, 1 = AI |
| `source` | str | Origin of the entry |
            """)

    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", "790")
        c2.metric("Human", "390")
        c3.metric("AI", "400")

        col_table, col_chart = st.columns([3, 2], gap="large")

        with col_table:
            st.subheader("AI Models Used")
            ai_df = pd.DataFrame(
                {
                    "Model": [
                        "openai/gpt-oss-120b",
                        "qwen/qwen2.5-7b-instruct",
                        "meta/llama-3.1-70b-instruct",
                        "mistralai/mixtral-8x22b-instruct-v0.1",
                    ],
                    "Entries": [100, 100, 100, 100],
                }
            )
            st.dataframe(ai_df, use_container_width=True, hide_index=True)

        with col_chart:
            st.subheader("Distribution")
            ag_chart = pd.DataFrame(
                {
                    "Source": ["Human", "gpt-oss-120b", "qwen2.5-7b", "llama-3.1-70b", "mixtral-8x22b"],
                    "Entries": [390, 100, 100, 100, 100],
                }
            ).set_index("Source")
            st.bar_chart(ag_chart, height=250, color="#ff4b4b")

        st.divider()

        st.subheader("Sample Data")
        st.caption(f"Showing: labels={label_f}, models={model_f}, limit={max_rows}")
        ag_df = load_ag_data()
        filtered_ag = ag_df[ag_df["label"].isin(label_f)]
        if "AI" in label_f and model_f:
            filtered_ag = filtered_ag[
                (filtered_ag["label"] == "Human") | (filtered_ag["source_model"].isin(model_f))
            ]
        st.dataframe(filtered_ag.head(max_rows), use_container_width=True, hide_index=True)

        with st.expander("Schema"):
            st.markdown("""
| Field | Type | Description |
|---|---|---|
| `id` | str | URL-friendly slug |
| `name` | str | Listing name |
| `description` | str | Full description |
| `label` | str | Human or AI |
| `source_model` | str | Generation model (AI only) |
            """)

    st.divider()

    st.subheader("Text Statistics")
    ts1, ts2, ts3, ts4 = st.columns(4)
    ts1.metric("Avg Words (Human)", "147")
    ts2.metric("Avg Words (AI)", "183")
    ts3.metric("Avg Sent. Len (Human)", "18.2")
    ts4.metric("Avg Sent. Len (AI)", "22.7")


def page_about():
    with st.sidebar:
        st.subheader("Links")
        st.caption("CSE 881 — Spring 2025")
        st.markdown("**Team size:** 4")

    st.header("About")
    st.caption("Automated Classification of Human vs AI Postings — CSE 881")

    st.markdown("""
    This project develops a system to classify and detect AI-generated text across online
    postings. We scrape real human data from job boards and farming directories, generate
    synthetic AI text from multiple LLMs, then train and evaluate a suite of classifiers.
    """)

    st.divider()

    st.subheader("Methodology")
    tab_collect, tab_preprocess, tab_models, tab_eval = st.tabs(
        [
            "Collection",
            "Preprocessing",
            "Models",
            "Evaluation",
        ]
    )

    with tab_collect:
        st.markdown("""
**Job postings** — ~1,000 real postings scraped from Indeed via Octoparse.
~1,000 AI postings generated across Claude, ChatGPT, Copilot, Gemini, and Perplexity.

**Agricultural listings** — 390 real listings scraped from the Care Farming Network
using Playwright + BeautifulSoup. 400 AI listings generated via 4 NVIDIA NIM models.
        """)

    with tab_preprocess:
        st.markdown("""
- HTML / noise removal, markdown stripping
- Salary normalization (hourly to annual), location formatting (City, State)
- Stopword removal, lowercasing, n-gram extraction (NLTK)
- Binary encoding (Human = 0, AI = 1)
- 80 / 10 / 10 stratified train / validation / test split
        """)

    with tab_models:
        model_df = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "Random Forest",
                    "SVM",
                    "CatBoost",
                    "LSTM",
                    "BERT",
                ],
                "Category": [
                    "Baseline",
                    "Baseline",
                    "Baseline",
                    "Gradient Boosting",
                    "Deep Learning",
                    "Deep Learning",
                ],
                "Description": [
                    "Linear classifier on TF-IDF features",
                    "Ensemble of decision trees on TF-IDF features",
                    "SVM with linear kernel on TF-IDF features",
                    "Handles categorical text features natively",
                    "LSTM network on tokenized text sequences",
                    "Fine-tuned BERT (bert_tiny_en_uncased)",
                ],
            }
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    with tab_eval:
        st.markdown("""
- Accuracy, precision, recall, F1, and AUC-ROC
- Cross-dataset evaluation (train on one domain, test on the other)
- Baseline vs. custom algorithm comparison
        """)

    st.divider()

    st.subheader("Team")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("**Hussain Aljafer**")
        st.caption(
            "Data collection, preprocessing, baseline models, AI detection algorithm"
        )
    with t2:
        st.markdown("**Wahid Hashem**")
        st.caption(
            "AI data generation, preprocessing, Streamlit app design & development"
        )
    with t3:
        st.markdown("**Aryan Sharma**")
        st.caption("AI data generation, baseline model research, evaluation, report")
    with t4:
        st.markdown("**Ricky Li**")
        st.caption("Agricultural scraping & generation, neural networks, evaluation")

    st.divider()

    st.subheader("Tech Stack")
    st.markdown("""
| Layer | Tools |
|---|---|
| Data & ML | Python, pandas, NumPy, scikit-learn, CatBoost, NLTK |
| Deep Learning | PyTorch / TensorFlow, text embeddings |
| Frontend | Streamlit, Plotly, Matplotlib |
| Scraping | Octoparse, Playwright, BeautifulSoup |
    """)


# Configure Streamlit App

st.set_page_config(
    page_title="AI Text Detector",
    layout="wide",
)

nav = st.navigation(
    [
        st.Page(page_home, title="Home", icon=":material/home:", url_path="home", default=True),
        st.Page(page_detector, title="Detector", icon=":material/search:", url_path="detector"),
        st.Page(page_performance, title="Performance", icon=":material/analytics:", url_path="performance"),
        st.Page(page_data, title="Data", icon=":material/database:", url_path="data"),
        st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
    ],
    position="top",
)

nav.run()
