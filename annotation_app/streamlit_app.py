"""Private Streamlit interface for expert arch/whorl subtype review."""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_app.app_logic import (  # noqa: E402
    CONFIDENCE_VALUES,
    build_export_csv,
    filtered_items,
    progress_counts,
    validate_annotation,
)
from annotation_app.backend import (  # noqa: E402
    AnnotationBackend,
    AnnotationBackendError,
    LocalAnnotationBackend,
    SupabaseAnnotationBackend,
)


st.set_page_config(
    page_title="Fingerprint Subtype Review",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #16332e;
        --muted: #60716d;
        --green: #176b5b;
        --green-soft: #e9f5f1;
        --amber-soft: #fff6df;
        --line: #dce7e3;
    }
    .stApp { background: #f7faf9; }
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #eef5f2; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 4px 18px rgba(22, 51, 46, 0.035);
    }
    div[data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid var(--line);
        box-shadow: 0 12px 34px rgba(22, 51, 46, 0.09);
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.25rem 1.35rem 1.4rem;
        box-shadow: 0 10px 30px rgba(22, 51, 46, 0.06);
    }
    div[role="radiogroup"] label {
        background: #f8fbfa;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.35rem;
    }
    div[role="radiogroup"] label:hover { border-color: var(--green); background: var(--green-soft); }
    .review-kicker {
        display: inline-block;
        color: var(--green);
        background: var(--green-soft);
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .instruction {
        color: var(--muted);
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }
    .type-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #153f37, #1f7464);
        color: white;
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }
    .type-banner strong { font-size: 1.15rem; }
    .type-banner span { opacity: 0.82; font-size: 0.85rem; }
    .login-copy { text-align: center; margin: 3rem auto 1.25rem; max-width: 620px; }
    .login-copy h1 { margin-bottom: 0.35rem; }
    .login-copy p { color: var(--muted); font-size: 1.02rem; }
    .local-notice {
        background: var(--green-soft);
        border: 1px solid #cfe6de;
        border-radius: 12px;
        color: var(--ink);
        padding: 0.7rem 0.9rem;
        margin: 0.5rem 0 1rem;
    }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        border-radius: 10px;
        min-height: 2.7rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_value(value: str) -> str:
    return "Select…" if value == "" else value.replace("_", " ").title()


def secret_section(name: str) -> dict[str, Any]:
    try:
        return dict(st.secrets[name])
    except (FileNotFoundError, KeyError):
        return {}


def runtime_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge ignored/hosted secrets with optional server environment values."""
    app_config = secret_section("app")
    supabase_config = secret_section("supabase")

    environment_app = {
        "backend": os.environ.get("ANNOTATION_BACKEND"),
        "review_access_code": os.environ.get("REVIEW_ACCESS_CODE"),
    }
    for name, value in environment_app.items():
        if value:
            app_config[name] = value

    if allowed := os.environ.get("ALLOWED_REVIEWER_IDS"):
        app_config["allowed_reviewer_ids"] = [
            value.strip() for value in allowed.split(",") if value.strip()
        ]

    environment_supabase = {
        "url": os.environ.get("SUPABASE_URL"),
        "service_role_key": os.environ.get("SUPABASE_SECRET_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        "bucket": os.environ.get("SUPABASE_BUCKET"),
        "export_path": os.environ.get("SUPABASE_EXPORT_PATH"),
    }
    for name, value in environment_supabase.items():
        if value:
            supabase_config[name] = value

    return app_config, supabase_config


@st.cache_resource(show_spinner=False)
def create_supabase_backend(
    url: str,
    service_role_key: str,
    bucket: str,
    export_path: str,
) -> SupabaseAnnotationBackend:
    return SupabaseAnnotationBackend(url, service_role_key, bucket, export_path)


@st.cache_resource(show_spinner=False)
def create_local_backend(
    package_dir: str, state_dir: str
) -> LocalAnnotationBackend:
    return LocalAnnotationBackend(Path(package_dir), Path(state_dir))


def require_reviewer(app_config: dict[str, Any]) -> str:
    """Require OIDC or the second-layer project access code."""
    allowed_reviewers = {
        str(value).strip().lower()
        for value in app_config.get("allowed_reviewer_ids", [])
        if str(value).strip()
    }
    use_oidc = bool(app_config.get("use_oidc", False))

    if use_oidc:
        if not st.user.is_logged_in:
            st.title("Private fingerprint subtype review")
            st.info("Sign in with the approved project account to continue.")
            st.button("Sign in", on_click=st.login, type="primary")
            st.stop()
        reviewer_id = str(
            st.user.get("email") or st.user.get("name") or "authenticated-reviewer"
        ).strip()
        if allowed_reviewers and reviewer_id.lower() not in allowed_reviewers:
            st.error("This authenticated account is not on the approved reviewer list.")
            st.button("Sign out", on_click=st.logout)
            st.stop()
        return reviewer_id

    if st.session_state.get("review_access_granted"):
        return str(st.session_state["reviewer_id"])

    st.markdown(
        """
        <div class="login-copy">
            <span class="review-kicker">Local research workspace</span>
            <h1>Fingerprint subtype review</h1>
            <p>Label one fingerprint at a time in a focused, private workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            '<div class="local-notice">🔒 Local preview — images and labels stay on this computer.</div>',
            unsafe_allow_html=True,
        )
        with st.form("reviewer_access_form"):
            reviewer_id = st.text_input("Reviewer ID", max_chars=80)
            access_code = st.text_input("Access code", type="password")
            submitted = st.form_submit_button(
                "Open review workspace", type="primary", width="stretch"
            )

    if submitted:
        expected_code = str(app_config.get("review_access_code", ""))
        clean_reviewer = reviewer_id.strip()
        reviewer_allowed = not allowed_reviewers or clean_reviewer.lower() in allowed_reviewers
        if (
            clean_reviewer
            and expected_code
            and reviewer_allowed
            and hmac.compare_digest(access_code, expected_code)
        ):
            st.session_state["review_access_granted"] = True
            st.session_state["reviewer_id"] = clean_reviewer
            st.rerun()
        else:
            st.error("The reviewer ID or project access code is not authorized.")
    st.stop()


def load_shared_data(
    backend: AnnotationBackend,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    try:
        items = backend.list_review_items()
        annotations = backend.list_annotations()
    except AnnotationBackendError as exc:
        st.error(str(exc))
        st.stop()
    return items, {str(row["review_id"]): row for row in annotations}


def get_private_image(
    backend: AnnotationBackend, review_id: str, image_path: str
) -> bytes:
    cache = st.session_state.setdefault("private_image_cache", {})
    if review_id not in cache:
        try:
            cache[review_id] = backend.download_image(image_path)
        except AnnotationBackendError as exc:
            st.error(str(exc))
            st.stop()
        while len(cache) > 5:
            cache.pop(next(iter(cache)))
    return cache[review_id]


app_config, configured_supabase = runtime_config()
reviewer_id = require_reviewer(app_config)
backend_mode = str(app_config.get("backend", "supabase")).strip().lower()
if backend_mode == "local":
    package_dir = Path(
        str(
            app_config.get(
                "local_package_dir",
                "data/processed/unlabeled_subtype_review_package_2026-08-04",
            )
        )
    )
    state_dir = Path(
        str(app_config.get("local_state_dir", "annotation_exports/local_preview"))
    )
    if not package_dir.is_absolute():
        package_dir = ROOT / package_dir
    if not state_dir.is_absolute():
        state_dir = ROOT / state_dir
    try:
        backend: AnnotationBackend = create_local_backend(
            str(package_dir.resolve()), str(state_dir.resolve())
        )
    except AnnotationBackendError as exc:
        st.error(str(exc))
        st.stop()
elif backend_mode == "supabase":
    supabase_config = configured_supabase
    required_supabase = {"url", "service_role_key", "bucket"}
    missing_config = sorted(required_supabase - set(supabase_config))
    if missing_config:
        st.error(
            "The app backend is not configured. Missing Streamlit secrets: "
            + ", ".join(f"supabase.{name}" for name in missing_config)
        )
        st.stop()
    backend = create_supabase_backend(
        str(supabase_config["url"]),
        str(supabase_config["service_role_key"]),
        str(supabase_config["bucket"]),
        str(
            supabase_config.get(
                "export_path", "exports/subtype_labeling_latest.csv"
            )
        ),
    )
else:
    st.error("app.backend must be either 'local' or 'supabase'.")
    st.stop()
items, annotations_by_id = load_shared_data(backend)

if not items:
    st.error("No review items are available. Run the secure package upload utility first.")
    st.stop()

counts = progress_counts(items, annotations_by_id)
st.markdown(
    '<span class="review-kicker">Local annotation workspace</span>',
    unsafe_allow_html=True,
)
st.title("Fingerprint subtype review")
st.caption(
    "Choose the subtype and confidence. Only add a note when something needs attention. · "
    + (
        "Nothing leaves this computer"
        if backend_mode == "local"
        else "annotations saved centrally"
    )
)

summary_columns = st.columns(4)
summary_columns[0].metric("Progress", f"{counts['completed']} / {counts['total']}")
summary_columns[1].metric("Remaining", counts["pending"])
summary_columns[2].metric(
    "Arch images", sum(item["current_main_type"] == "arch" for item in items)
)
summary_columns[3].metric(
    "Whorl images", sum(item["current_main_type"] == "whorl" for item in items)
)

if flash_message := st.session_state.pop("flash_message", None):
    st.success(flash_message)
if csv_warning := st.session_state.pop("csv_warning", None):
    st.warning(csv_warning)

with st.sidebar:
    st.subheader("Review progress")
    st.progress(counts["completed"] / counts["total"] if counts["total"] else 0)
    metric_columns = st.columns(2)
    metric_columns[0].metric("Completed", counts["completed"])
    metric_columns[1].metric("Pending", counts["pending"])
    st.caption(
        f"{counts['warnings']} alternative-code warnings · "
        f"{counts['adjudicate']} awaiting adjudication · {counts['excluded']} excluded"
    )

    st.divider()
    st.subheader("Filter queue")
    main_type_filter = st.selectbox(
        "Main type", ["all", "arch", "whorl"], format_func=display_value
    )
    status_filter = st.selectbox(
        "Status", ["pending", "completed", "all"], format_func=display_value
    )
    warnings_only = st.checkbox("Alternative-code warnings only")

    export_bytes = build_export_csv(items, annotations_by_id)
    st.download_button(
        "Download current CSV",
        data=export_bytes,
        file_name="subtype_labeling_latest.csv",
        mime="text/csv",
        width="stretch",
    )
    if st.button("Refresh labels", width="stretch"):
        st.session_state.pop("private_image_cache", None)
        st.rerun()

    st.divider()
    st.caption(f"Reviewer: {reviewer_id}")
    if app_config.get("use_oidc", False):
        st.button("Sign out", on_click=st.logout, width="stretch")
    elif st.button("End review session", width="stretch"):
        st.session_state.clear()
        st.rerun()

queue = filtered_items(
    items,
    annotations_by_id,
    main_type=main_type_filter,
    status=status_filter,
    warnings_only=warnings_only,
)
if not queue:
    st.info("No images match the current filters.")
    st.stop()

queue_ids = [str(item["review_id"]) for item in queue]
current_id = str(st.session_state.get("current_review_id", queue_ids[0]))
if current_id not in queue_ids:
    current_id = queue_ids[0]
    st.session_state["current_review_id"] = current_id

selected_id = st.selectbox(
    "Review item",
    queue_ids,
    index=queue_ids.index(current_id),
    label_visibility="collapsed",
)
if selected_id != current_id:
    current_id = selected_id
    st.session_state["current_review_id"] = current_id

position = queue_ids.index(current_id)
current_item = next(item for item in queue if str(item["review_id"]) == current_id)
existing = annotations_by_id.get(current_id, {})

nav_left, nav_status, nav_right = st.columns([1, 2, 1])
if nav_left.button("← Previous", disabled=position == 0, width="stretch"):
    st.session_state["current_review_id"] = queue_ids[position - 1]
    st.rerun()
nav_status.markdown(
    f"<div style='text-align:center;padding-top:0.45rem'>Queue item "
    f"{position + 1} of {len(queue)}</div>",
    unsafe_allow_html=True,
)
if nav_right.button(
    "Next →", disabled=position == len(queue) - 1, width="stretch"
):
    st.session_state["current_review_id"] = queue_ids[position + 1]
    st.rerun()

image_column, form_column = st.columns([1.05, 1], gap="large")
with image_column:
    st.image(
        get_private_image(
            backend, current_id, str(current_item["image_path"])
        ),
        caption=f"Review ID {current_id}",
        width="stretch",
    )
    if current_item.get("alternative_type_warning"):
        st.error(
            "Alternative examiner code(s) span more than one main type. "
            "Do not force a subtype if the displayed main type is doubtful."
        )
    with st.expander("Record details"):
        st.write("Current main type:", current_item["current_main_type"])
        st.write("Source primary code:", current_item["source_primary_code"])
        st.write("Recorded pattern codes:", current_item["recorded_pattern_codes"])

with form_column:
    st.markdown(
        f"""
        <div class="type-banner">
            <span>Current main type</span>
            <strong>{str(current_item['current_main_type']).upper()}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    subtype_options = list(current_item["permitted_subtypes"])
    confidence_options = list(CONFIDENCE_VALUES)
    existing_subtype = str(existing.get("confirmed_subtype") or "")
    existing_confidence = str(existing.get("confidence") or "")
    subtype_index = (
        subtype_options.index(existing_subtype)
        if existing_subtype in subtype_options
        else None
    )
    confidence_index = (
        confidence_options.index(existing_confidence)
        if existing_confidence in confidence_options
        else None
    )

    with st.form(f"annotation_form_{current_id}", clear_on_submit=False):
        st.subheader("1. Choose the subtype")
        st.markdown(
            '<p class="instruction">Select the most specific pattern you can defend. Choose “Unclear” instead of guessing.</p>',
            unsafe_allow_html=True,
        )
        confirmed_subtype = st.radio(
            "Confirmed subtype",
            subtype_options,
            index=subtype_index,
            format_func=display_value,
            label_visibility="collapsed",
        )
        st.divider()
        st.subheader("2. Confidence")
        confidence = st.radio(
            "Confidence",
            confidence_options,
            index=confidence_index,
            format_func=display_value,
            horizontal=True,
            label_visibility="collapsed",
        )
        with st.expander("Flag a concern or add a note"):
            main_type_concern = st.checkbox(
                "The current main type may be wrong",
                value=bool(existing.get("main_type_issue")),
            )
            review_notes = st.text_area(
                "Note",
                value=str(existing.get("review_notes") or ""),
                height=100,
                placeholder="Explain why this image needs another look.",
            )
            st.caption(
                "A note is required only when choosing Unclear or flagging the main type."
            )
        submitted = st.form_submit_button(
            "Save label and continue →", type="primary", width="stretch"
        )

    if submitted:
        main_type_issue = "uncertain" if main_type_concern else ""
        review_action = (
            "adjudicate"
            if confirmed_subtype == "unclear" or main_type_concern
            else "accept"
        )
        proposed = {
            "confirmed_subtype": confirmed_subtype or "",
            "confidence": confidence or "",
            "review_action": review_action,
            "main_type_issue": main_type_issue,
            "review_notes": review_notes.strip(),
            "reviewer_id": reviewer_id,
        }
        errors = validate_annotation(current_item, proposed)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                saved = backend.save_annotation(
                    current_id,
                    proposed,
                    int(existing.get("revision") or 0),
                )
                fresh_annotations = backend.list_annotations()
                fresh_by_id = {
                    str(row["review_id"]): row for row in fresh_annotations
                }
                try:
                    backend.write_latest_csv(build_export_csv(items, fresh_by_id))
                except AnnotationBackendError as exc:
                    st.session_state["csv_warning"] = str(exc)

                remaining = filtered_items(
                    items,
                    fresh_by_id,
                    main_type=main_type_filter,
                    status="pending",
                    warnings_only=warnings_only,
                )
                remaining_ids = [str(item["review_id"]) for item in remaining]
                if remaining_ids:
                    st.session_state["current_review_id"] = remaining_ids[0]
                st.session_state["flash_message"] = (
                    f"Saved {current_id} as {saved.get('confirmed_subtype', confirmed_subtype)}."
                )
                st.rerun()
            except AnnotationBackendError as exc:
                st.error(str(exc))
