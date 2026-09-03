import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="SIP Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "SIP_FAQ_Bot_Database_Complete.xlsx"

SIMILARITY_THRESHOLD = 0.12


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def load_faq_database(file_path_string, file_modified_time):
    file_path = Path(file_path_string)

    if not file_path.exists():
        return None

    dataframe = pd.read_excel(file_path)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    dataframe = dataframe.fillna("")

    required_columns = [
        "FAQ_ID",
        "Question",
        "Proposed_Answer",
        "Relevant_Policy_or_Source",
        "Source_Status",
        "Keywords",
        "Record_Status",
    ]

    for column in required_columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[
        dataframe["Record_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("published")
    ].copy()

    dataframe["Search_Text"] = (
        dataframe["Question"].astype(str)
        + " "
        + dataframe["Keywords"]
        .astype(str)
        .str.replace(";", " ", regex=False)
    )

    return dataframe


def keyword_overlap_score(user_question, keywords):
    user_words = set(clean_text(user_question).split())
    keyword_words = set(clean_text(keywords).split())

    if not user_words or not keyword_words:
        return 0.0

    shared_words = user_words.intersection(keyword_words)
    return len(shared_words) / len(user_words)


def find_best_faq(user_question, faq_database):
    cleaned_question = clean_text(user_question)
    searchable_texts = faq_database["Search_Text"].apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    vectors = vectorizer.fit_transform(searchable_texts + [cleaned_question])

    similarity_scores = cosine_similarity(
        vectors[-1],
        vectors[:-1],
    ).flatten()

    keyword_scores = faq_database["Keywords"].apply(
        lambda keywords: keyword_overlap_score(cleaned_question, keywords)
    ).to_numpy()

    combined_scores = (similarity_scores * 0.75) + (keyword_scores * 0.25)

    best_index = combined_scores.argmax()

    return faq_database.iloc[best_index], float(combined_scores[best_index])


def build_bot_reply(faq_record):
    answer = str(faq_record["Proposed_Answer"]).strip()

    if not answer:
        answer = (
            "I found a related FAQ record, but its answer is incomplete. "
            "Please check SIP Teams or contact your Learning Officer (LO)."
        )

    return answer


def stream_reply(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.016)


def initialise_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I’m your **SIP Knowledge Assistant**. 👋\n\n"
                "Ask me naturally about your Student Internship Programme—"
                "for example, MC submission, attendance, learning journals, "
                "workplace issues, overtime, injuries, dress code, or the SIP duration."
            ),
            "metadata": None,
        }
    ]


def refresh_database():
    load_faq_database.clear()
    st.session_state.database_refresh_notice = True


def clear_chat():
    initialise_chat()


def ask_quick_question(question):
    st.session_state.pending_question = question


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            background: linear-gradient(135deg, #0b3d91 0%, #1463d8 100%);
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 20px rgba(11, 61, 145, 0.18);
        }

        .app-header h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
        }

        .app-header p {
            margin: 0.45rem 0 0 0;
            opacity: 0.94;
            font-size: 1rem;
        }

        .info-card {
            background: #f5f8ff;
            border: 1px solid #dbe7ff;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .small-note {
            color: #5b6472;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>🎓 SIP Knowledge Assistant</h1>
        <p>A conversational guide for Student Internship Programme FAQs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATABASE_PATH.exists():
    st.error(
        "I cannot locate the FAQ database file. Confirm that "
        "`SIP_FAQ_Bot_Database_Complete.xlsx` is in the same GitHub folder as `app.py`."
    )
    st.stop()

database_modified_time = DATABASE_PATH.stat().st_mtime
faq_database = load_faq_database(
    str(DATABASE_PATH),
    database_modified_time,
)

if faq_database is None or faq_database.empty:
    st.error(
        "The FAQ database could not be loaded, or it contains no rows marked `Published`."
    )
    st.stop()

if "messages" not in st.session_state:
    initialise_chat()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "database_refresh_notice" not in st.session_state:
    st.session_state.database_refresh_notice = False

with st.sidebar:
    st.subheader("Assistant controls")

    if st.button("🔄 Refresh knowledge base", use_container_width=True):
        refresh_database()
        st.rerun()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        clear_chat()
        st.rerun()

    st.divider()

    st.metric(
        label="Published FAQ records",
        value=len(faq_database),
    )

    st.caption("Database file")
    st.code("SIP_FAQ_Bot_Database_Complete.xlsx", language=None)

    st.caption(
        "Use **Refresh knowledge base** after uploading or replacing the Excel file in GitHub."
    )

    st.divider()

    st.subheader("Coverage")
    st.caption(
        "MCs • schedules • journals • attendance • workplace conduct • "
        "injuries • overtime • social media • SIP duration"
    )

if st.session_state.database_refresh_notice:
    st.success(
        "Knowledge base refreshed. The assistant is now using the latest Excel file."
    )
    st.session_state.database_refresh_notice = False

st.markdown(
    """
    <div class="info-card">
        <strong>Ask freely.</strong> You do not need to use the exact wording from the FAQ.
        Try: “How many weeks is my internship?”, “Where do I send my doctor note?”,
        or “I might be late because of traffic.”
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Quick questions")

quick_col_1, quick_col_2, quick_col_3, quick_col_4 = st.columns(4)

with quick_col_1:
    if st.button("How do I submit my MC?", use_container_width=True):
        ask_quick_question("How do I submit my MC?")

with quick_col_2:
    if st.button("What if I am late?", use_container_width=True):
        ask_quick_question("What should I do if I am late for work?")

with quick_col_3:
    if st.button("How long is SIP?", use_container_width=True):
        ask_quick_question("How long is SIP?")

with quick_col_4:
    if st.button("What if I get injured?", use_container_width=True):
        ask_quick_question("What should I do if I get injured at work?")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("metadata"):
            metadata = message["metadata"]

            with st.expander("Reference details"):
                st.write(f"**FAQ ID:** {metadata['faq_id']}")
                st.write(f"**Source:** {metadata['source']}")
                st.write(f"**Source status:** {metadata['source_status']}")
                st.write(
                    f"**Match confidence:** {metadata['confidence']:.0%}"
                )

user_question = st.chat_input(
    "Ask a question about SIP...",
)

if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
            "metadata": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the SIP knowledge base..."):
            best_faq, confidence_score = find_best_faq(
                user_question,
                faq_database,
            )

        if confidence_score < SIMILARITY_THRESHOLD:
            reply = (
                "I’m sorry, but I could not find a reliable answer for that in the "
                "current SIP FAQ database.\n\n"
                "Please check **SIP Teams** or contact your **Learning Officer (LO)** "
                "for clarification."
            )
            metadata = None
        else:
            reply = build_bot_reply(best_faq)
            metadata = {
                "faq_id": str(best_faq["FAQ_ID"]).strip(),
                "source": str(
                    best_faq["Relevant_Policy_or_Source"]
                ).strip(),
                "source_status": str(best_faq["Source_Status"]).strip(),
                "confidence": confidence_score,
            }

        completed_reply = st.write_stream(
            stream_reply(reply),
            cursor="▌",
        )

        if metadata:
            with st.expander("Reference details"):
                st.write(f"**FAQ ID:** {metadata['faq_id']}")
                st.write(f"**Source:** {metadata['source']}")
                st.write(f"**Source status:** {metadata['source_status']}")
                st.write(
                    f"**Match confidence:** {metadata['confidence']:.0%}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": completed_reply,
            "metadata": metadata,
        }
    )

st.markdown(
    """
    <p class="small-note">
        This assistant provides answers from the uploaded SIP FAQ database.
        For matters requiring confirmation, consult the latest SIP Teams updates
        or your Learning Officer.
    </p>
    """,
    unsafe_allow_html=True,
)
