"""Local Streamlit interface for broad fingerprint-pattern classification."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import sys
import time

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broad_classifier.app_logic import assess_prediction  # noqa: E402
from broad_classifier.inference import (  # noqa: E402
    CLASS_NAMES,
    DISPLAY_NAMES,
    BroadPatternEnsemble,
    InputImageError,
    PredictionResult,
    decode_image,
)
from broad_classifier.model_assets import (  # noqa: E402
    CHECKPOINT_ASSETS,
    ensure_checkpoints,
)


DEFAULT_MODEL_DIRECTORY = ROOT / "models" / "efficientnet_320_cv"
MODEL_DIRECTORY = Path(
    os.environ.get("BROAD_CLASSIFIER_MODEL_DIR", str(DEFAULT_MODEL_DIRECTORY))
)

PATTERN_DESCRIPTIONS = {
    "arch": (
        "Ridges generally flow from one side to the other with a central rise. "
        "Plain and tented arch subtypes are not separated by this model."
    ),
    "left_slant_loop": (
        "A recurving loop assigned to the left-slant category under the dataset's "
        "annotation convention. Mirroring an image can change loop direction."
    ),
    "right_slant_loop": (
        "A recurving loop assigned to the right-slant category under the dataset's "
        "annotation convention. Mirroring an image can change loop direction."
    ),
    "whorl": (
        "A circular, spiral, or more complex recurving ridge formation. Individual "
        "whorl subtypes are not separated by this model."
    ),
}


st.set_page_config(
    page_title="Fingerprint Pattern Classifier",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #142f2a;
        --muted: #62736f;
        --green: #176c5b;
        --green-dark: #0e4439;
        --green-soft: #e8f5f0;
        --blue-soft: #eaf2f8;
        --amber: #9a6009;
        --amber-soft: #fff4dc;
        --line: #d8e5e0;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 3%, rgba(43, 137, 115, 0.08), transparent 26rem),
            #f6f9f8;
    }
    .block-container { max-width: 1280px; padding-top: 1.45rem; padding-bottom: 3.5rem; }
    [data-testid="stSidebar"] { background: #edf4f1; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(130deg, #103b33 0%, #196957 66%, #2b8a73 100%);
        border-radius: 24px;
        color: white;
        padding: 2rem 2.1rem 1.85rem;
        box-shadow: 0 18px 48px rgba(16, 59, 51, 0.18);
        margin-bottom: 1.15rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -115px;
        border: 32px solid rgba(255,255,255,0.075);
        border-radius: 50%;
    }
    .hero .eyebrow {
        display: inline-block;
        color: #c3eadf;
        font-size: 0.73rem;
        font-weight: 760;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero h1 { color: white; margin: 0 0 0.45rem; font-size: 2.2rem; }
    .hero p { color: #e0f2ed; margin: 0; max-width: 760px; font-size: 1.02rem; }
    .hero-stats { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.05rem; }
    .hero-stat {
        background: rgba(255,255,255,0.105);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 999px;
        color: #f2fbf8;
        padding: 0.34rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 650;
    }
    .workflow {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 0.2rem 0 1.05rem;
    }
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.72rem 0.85rem;
        color: var(--muted);
    }
    .workflow-step.active { border-color: #83bcae; background: var(--green-soft); color: var(--ink); }
    .workflow-step.done { border-color: #b7d9cf; color: var(--green-dark); }
    .step-number {
        display: grid;
        place-items: center;
        width: 1.8rem;
        height: 1.8rem;
        flex: 0 0 1.8rem;
        border-radius: 50%;
        background: #edf2f0;
        color: var(--ink);
        font-size: 0.8rem;
        font-weight: 780;
    }
    .active .step-number, .done .step-number { background: var(--green); color: white; }
    .step-copy strong { display: block; font-size: 0.85rem; }
    .step-copy span { display: block; font-size: 0.73rem; margin-top: 0.05rem; }
    .privacy-strip {
        background: var(--green-soft);
        border: 1px solid #c8e2d8;
        border-radius: 13px;
        color: var(--ink);
        padding: 0.72rem 0.9rem;
        margin: 0.15rem 0 1.1rem;
    }
    .upload-panel, .result-card, .explanation-card, .empty-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 19px;
        box-shadow: 0 10px 30px rgba(20, 47, 42, 0.055);
    }
    .upload-panel { padding: 1rem 1.1rem 0.25rem; margin-bottom: 1rem; }
    .result-card { padding: 1.35rem 1.45rem; margin-bottom: 1rem; }
    .explanation-card { padding: 1.05rem 1.15rem; margin: 0.7rem 0; }
    .empty-card { padding: 1.1rem 1.2rem; color: var(--muted); margin-top: 1rem; }
    .result-label, .card-kicker {
        color: var(--green);
        font-size: 0.76rem;
        font-weight: 760;
        letter-spacing: 0.075em;
        text-transform: uppercase;
    }
    .result-name { color: var(--ink); font-size: 2.25rem; font-weight: 780; line-height: 1.13; }
    .result-score { color: var(--muted); margin-top: 0.28rem; }
    .result-summary { color: var(--muted); line-height: 1.55; margin-top: 0.8rem; }
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin: 0.8rem 0 1rem;
    }
    .detail-chip {
        background: #f8fbfa;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.62rem 0.72rem;
    }
    .detail-chip span { display: block; color: var(--muted); font-size: 0.7rem; text-transform: uppercase; }
    .detail-chip strong { color: var(--ink); font-size: 0.9rem; }
    .fold-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.55rem; }
    .fold-card {
        background: #f8fbfa;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 0.75rem 0.65rem;
        text-align: center;
    }
    .fold-card.winner { background: var(--green-soft); border-color: #abd4c7; }
    .fold-card span { display: block; color: var(--muted); font-size: 0.7rem; }
    .fold-card strong { display: block; color: var(--ink); font-size: 0.82rem; margin-top: 0.2rem; }
    .class-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.65rem; }
    .class-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.85rem;
    }
    .class-card strong { color: var(--ink); display: block; font-size: 0.86rem; }
    .class-card span { color: var(--muted); display: block; font-size: 0.75rem; margin-top: 0.18rem; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 5px 16px rgba(20, 47, 42, 0.035);
    }
    div[data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 9px 25px rgba(20, 47, 42, 0.07);
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #9fc8bd;
        border-radius: 15px;
        padding: 0.55rem 0.8rem;
    }
    .stButton > button {
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
    }
    .method-note { color: var(--muted); font-size: 0.88rem; }
    @media (max-width: 800px) {
        .workflow, .class-grid { grid-template-columns: 1fr; }
        .fold-grid { grid-template-columns: repeat(2, 1fr); }
        .detail-grid { grid-template-columns: 1fr; }
        .hero { padding: 1.55rem 1.25rem; }
        .hero h1 { font-size: 1.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_ensemble(model_directory: str) -> BroadPatternEnsemble:
    """Retrieve, verify, and load the five CPU models once per server process."""
    verified_directory = ensure_checkpoints(Path(model_directory))
    return BroadPatternEnsemble.from_directory(verified_directory, device_name="cpu")


def clear_analysis() -> None:
    """Clear image-dependent state and rotate the uploader widget key."""
    for key in (
        "prediction_upload_digest",
        "prediction_result",
        "prediction_preview",
        "prediction_seconds",
    ):
        st.session_state.pop(key, None)
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1


def render_sidebar() -> None:
    checkpoint_count = sum(
        (MODEL_DIRECTORY / asset.filename).is_file()
        for asset in CHECKPOINT_ASSETS
    )
    st.sidebar.markdown("## Pattern classifier")
    if checkpoint_count == 5:
        st.sidebar.success("Five-model ensemble ready")
    else:
        st.sidebar.info("Five verified models download on the first analysis")

    st.sidebar.markdown("### Model profile")
    st.sidebar.markdown(
        "**Architecture:** EfficientNet-B0  \n"
        "**Evaluation:** Subject-grouped CV  \n"
        "**Input:** 320 × 320 CLAHE  \n"
        "**Classes:** Four broad patterns"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Responsible use")
    st.sidebar.info(
        "This model classifies ridge-pattern appearance. It does not identify a "
        "person and does not yet classify arch or whorl subtypes."
    )
    st.sidebar.warning(
        "Do not use the output as an autonomous forensic conclusion."
    )
    st.sidebar.caption("Private research prototype • Version 0.1")


def render_workflow(has_upload: bool, has_result: bool) -> None:
    states = (
        "done" if has_upload else "active",
        "done" if has_result else ("active" if has_upload else ""),
        "active" if has_result else "",
    )
    st.markdown(
        f"""
        <div class="workflow">
            <div class="workflow-step {states[0]}">
                <div class="step-number">1</div>
                <div class="step-copy"><strong>Upload</strong><span>Select one rolled print</span></div>
            </div>
            <div class="workflow-step {states[1]}">
                <div class="step-number">2</div>
                <div class="step-copy"><strong>Analyze</strong><span>Run five trained models</span></div>
            </div>
            <div class="workflow-step {states[2]}">
                <div class="step-number">3</div>
                <div class="step-copy"><strong>Interpret</strong><span>Review score and agreement</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probabilities(result: PredictionResult) -> None:
    ordered = sorted(
        result.class_probabilities.items(), key=lambda item: item[1], reverse=True
    )
    for class_name, probability in ordered:
        st.progress(
            probability,
            text=f"{DISPLAY_NAMES[class_name]} — {probability:.1%}",
        )


def render_fold_consensus(result: PredictionResult) -> None:
    cards = []
    for fold, class_name in enumerate(result.fold_predictions, start=1):
        winner_class = "winner" if class_name == result.predicted_class else ""
        cards.append(
            f'<div class="fold-card {winner_class}"><span>Fold {fold}</span>'
            f"<strong>{DISPLAY_NAMES[class_name]}</strong></div>"
        )
    st.markdown('<div class="fold-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


render_sidebar()

result = st.session_state.get("prediction_result")
preprocessed = st.session_state.get("prediction_preview")
has_result = isinstance(result, PredictionResult) and preprocessed is not None

st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">Dermatoglyphic research prototype</span>
        <h1>Broad fingerprint pattern classifier</h1>
        <p>Analyze one rolled fingerprint impression and review the ensemble's
        predicted broad ridge pattern, model probability, and internal agreement.</p>
        <div class="hero-stats">
            <span class="hero-stat">4 pattern classes</span>
            <span class="hero-stat">5-model ensemble</span>
            <span class="hero-stat">Subject-disjoint development</span>
            <span class="hero-stat">Transient image processing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="privacy-strip">
        <strong>Private local processing:</strong> image bytes are used for the
        current session, are not written to disk, and are not added to training data.
        The preview shown below is decoded without displaying the original filename.
    </div>
    """,
    unsafe_allow_html=True,
)

uploader_nonce = st.session_state.get("uploader_nonce", 0)
workflow_slot = st.empty()
uploaded_file = st.file_uploader(
    "Upload a rolled fingerprint image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="PNG, JPEG, or TIFF; maximum upload size 20 MB.",
    key=f"fingerprint_upload_{uploader_nonce}",
)

image_bytes: bytes | None = None
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    upload_digest = sha256(image_bytes).hexdigest()
    if st.session_state.get("prediction_upload_digest") != upload_digest:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_preview", None)
        st.session_state.pop("prediction_seconds", None)
        st.session_state["prediction_upload_digest"] = upload_digest

result = st.session_state.get("prediction_result")
preprocessed = st.session_state.get("prediction_preview")
has_result = isinstance(result, PredictionResult) and preprocessed is not None
with workflow_slot.container():
    render_workflow(uploaded_file is not None, has_result)

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-card">
            <div class="card-kicker">Before you begin</div>
            Upload a single rolled fingerprint with the complete central pattern area
            visible. Avoid cropped cores, latent marks, slap impressions, mirrored
            images, and photographs containing more than one finger.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Patterns the model can report")
    st.markdown(
        """
        <div class="class-grid">
            <div class="class-card"><strong>Arch</strong><span>Broad arch family</span></div>
            <div class="class-card"><strong>Left-slant loop</strong><span>Dataset direction convention</span></div>
            <div class="class-card"><strong>Right-slant loop</strong><span>Dataset direction convention</span></div>
            <div class="class-card"><strong>Whorl</strong><span>Broad whorl family</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        assert image_bytes is not None
        decoded_preview = decode_image(image_bytes)
        upload_error = None
    except InputImageError as exc:
        decoded_preview = None
        upload_error = str(exc)

    if upload_error:
        st.error(upload_error)
    else:
        height, width = decoded_preview.shape
        size_megabytes = len(image_bytes) / (1024 * 1024)
        preview_column, action_column = st.columns([1, 1.15], gap="large")
        with preview_column:
            st.markdown("#### Uploaded impression")
            st.image(decoded_preview, channels="GRAY", width="stretch")
        with action_column:
            st.markdown("#### Ready for analysis")
            st.markdown(
                f"""
                <div class="detail-grid">
                    <div class="detail-chip"><span>Dimensions</span><strong>{width} × {height} px</strong></div>
                    <div class="detail-chip"><span>File size</span><strong>{size_megabytes:.2f} MB</strong></div>
                    <div class="detail-chip"><span>Processing</span><strong>Memory only</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                "Confirm that the central ridge pattern is visible and that the "
                "image has not been horizontally mirrored. The model will crop "
                "white margins, enhance local contrast, and run five checkpoints."
            )
            analyze = st.button(
                "Analyze fingerprint",
                type="primary",
                width="stretch",
            )

        if analyze:
            try:
                start = time.perf_counter()
                with st.spinner(
                    "Preparing the image and loading five fold models. "
                    "The first analysis may take a minute…"
                ):
                    ensemble = load_ensemble(str(MODEL_DIRECTORY.resolve()))
                    result, preprocessed = ensemble.predict_bytes(image_bytes)
                st.session_state["prediction_result"] = result
                st.session_state["prediction_preview"] = preprocessed
                st.session_state["prediction_seconds"] = time.perf_counter() - start
                has_result = True
            except InputImageError as exc:
                st.error(str(exc))
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(
                    "The classifier could not be loaded or run. Ask the project "
                    f"administrator to verify the model files. Details: {exc}"
                )

result = st.session_state.get("prediction_result")
preprocessed = st.session_state.get("prediction_preview")
if isinstance(result, PredictionResult) and preprocessed is not None:
    st.markdown("---")
    overview_tab, probabilities_tab, ensemble_tab, method_tab = st.tabs(
        ["Overview", "Probabilities", "Ensemble details", "How it works"]
    )

    with overview_tab:
        result_column, processed_column = st.columns([1.25, 0.75], gap="large")
        with result_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted broad pattern</div>
                    <div class="result-name">{result.display_label}</div>
                    <div class="result-score">Mean ensemble probability:
                    <strong>{result.predicted_probability:.1%}</strong></div>
                    <div class="result-summary">{PATTERN_DESCRIPTIONS[result.predicted_class]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_one, metric_two, metric_three = st.columns(3)
            metric_one.metric("Model probability", f"{result.predicted_probability:.1%}")
            metric_two.metric("Fold agreement", f"{result.agreement:.0%}")
            metric_three.metric("Top-two margin", f"{result.top_two_margin:.1%}")

            assessment = assess_prediction(result)
            if assessment.needs_review:
                st.warning(
                    "Manual review recommended\n\n- " + "\n- ".join(assessment.reasons)
                )
            else:
                st.success("The five-model result is internally consistent.")
        with processed_column:
            st.markdown("#### Model input")
            st.image(
                preprocessed,
                clamp=True,
                channels="GRAY",
                width="stretch",
                caption="Cropped, CLAHE-enhanced, 320 × 320",
            )
            inference_seconds = st.session_state.get("prediction_seconds")
            if inference_seconds is not None:
                st.caption(f"Completed locally in {inference_seconds:.2f} seconds.")

        st.info(
            "Interpret the probability as a model score, not a guarantee or a "
            "calibrated statement of correctness. Consequential or uncertain cases "
            "require qualified human review."
        )

    with probabilities_tab:
        st.subheader("Probability distribution")
        st.write(
            "The ensemble averages the four-class softmax probabilities produced "
            "by all five checkpoints. The largest average determines the result."
        )
        render_probabilities(result)
        st.markdown(
            '<div class="explanation-card"><div class="card-kicker">Reading the margin</div>'
            "The top-two margin is the difference between the two leading average "
            "probabilities. A small margin indicates that the model sees competing "
            "pattern evidence.</div>",
            unsafe_allow_html=True,
        )

    with ensemble_tab:
        st.subheader("Five-fold consensus")
        st.write(
            "Every card below represents one independently trained checkpoint of "
            "the same EfficientNet-B0 architecture. Highlighted cards match the "
            "final ensemble prediction."
        )
        render_fold_consensus(result)
        st.markdown(
            f"""
            <div class="explanation-card">
                <div class="card-kicker">Consensus summary</div>
                <strong>{sum(name == result.predicted_class for name in result.fold_predictions)} of 5</strong>
                fold models selected {result.display_label}. The displayed probability
                is calculated from the full probability vectors, not from a simple vote.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with method_tab:
        st.subheader("From upload to prediction")
        st.markdown(
            """
            1. The uploaded image is validated and decoded as grayscale.
            2. Mostly white margins are detected and cropped with a safety margin.
            3. CLAHE enhances local ridge contrast before resizing.
            4. Aspect ratio is preserved on a white 320 × 320 canvas.
            5. The image is normalized using the ImageNet channel statistics used
               during model development.
            6. Five EfficientNet-B0 checkpoints produce four probabilities each.
            7. The app averages those probabilities and reports the leading class.
            """
        )
        st.warning(
            "The reported 91.73% development accuracy is grouped out-of-fold "
            "performance. It is not yet an independent accuracy measurement for "
            "this five-checkpoint ensemble or for arbitrary uploaded images."
        )

    st.button(
        "Clear result and analyze another fingerprint",
        on_click=clear_analysis,
        width="stretch",
    )
