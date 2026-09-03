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
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "SIP_FAQ_Bot_Database_Complete.xlsx"

SIMILARITY_THRESHOLD = 0.12


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data
def load_faq_database(file_path):
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
        dataframe["Record_Status"].astype(str).str.strip().str.lower()
        == "published"
    ].copy()

    dataframe["Search_Text"] = (
        dataframe["Question"].astype(str)
        + " "
        + dataframe["Keywords"]
        .astype(str)
        .str.replace(";", " ", regex=False)
    )

    return dataframe


def keyword_overlap_score(user_text, keyword_text):
    user_words = set(clean_text(user_text).split())
    keyword_words = set(clean_text(keyword_text).split())

    if not user_words or not keyword_words:
        return 0.0

    return len(user_words.intersection(keyword_words)) / len(user_words)


def find_best_faq(user_question, faq_database):
    cleaned_question = clean_text(user_question)
    searchable_text = faq_database["Search_Text"].apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    vectors = vectorizer.fit_transform(searchable_text + [cleaned_question])

    similarity_scores = cosine_similarity(
        vectors[-1],
        vectors[:-1],
    ).flatten()

    keyword_scores = faq_database["Keywords"].apply(
        lambda value: keyword_overlap_score(cleaned_question, value)
    ).to_numpy()

    combined_scores = (similarity_scores * 0.75) + (keyword_scores * 0.25)

    best_index = combined_scores.argmax()

    return faq_database.iloc[best_index], combined_scores[best_index]


def create_answer(faq_record):
    answer = str(faq_record["Proposed_Answer"]).strip()
    faq_id = str(faq_record["FAQ_ID"]).strip()
    source = str(faq_record["Relevant_Policy_or_Source"]).strip()
    source_status = str(faq_record["Source_Status"]).strip()

    response = answer

    if faq_id:
        response += f"\n\n**Reference:** {faq_id}"

    if source:
        response += f"\n\n**Source:** {source}"

    if source_status:
        response += f"\n\n**Source status:** {source_status}"

    return response


def generate_response(user_question, faq_database):
    faq_record, score = find_best_faq(user_question, faq_database)

    if score < SIMILARITY_THRESHOLD:
        return (
            "I’m sorry, but I could not find a reliable answer for that in the "
            "current SIP FAQ database.\n\n"
            "Please check the latest information on **SIP Teams** or contact your "
            "**Learning Officer (LO)** for clarification."
        )

    return create_answer(faq_record)


def type_response(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.018)


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I’m your SIP Knowledge Assistant. 👋\n\n"
                "Ask me about MC submission, attendance, learning journals, "
                "workplace concerns, dress code, injuries, overtime, schedules, "
                "or other SIP-related matters."
            ),
        }
    ]


faq_database = load_faq_database(DATABASE_FILE)

st.title("🎓 SIP Knowledge Assistant")
st.caption(
    "Ask SIP questions naturally. I will search the approved FAQ database."
)

if faq_database is None:
    st.error(
        "The FAQ Excel file could not be found. "
        "Make sure `SIP_FAQ_Bot_Database_Complete.xlsx` is uploaded beside `app.py`."
    )
    st.stop()

if "messages" not in st.session_state:
    reset_chat()

with st.sidebar:
    if st.button("🗑️ Clear chat"):
        reset_chat()
        st.rerun()

    st.caption(f"Loaded {len(faq_database)} published FAQ records.")

    with st.expander("Example questions"):
        st.write(
            "- I am sick. Where do I submit my MC?\n"
            "- I may be late because of traffic.\n"
            "- Can I work from home during SIP?\n"
            "- What happens if I get injured at work?\n"
            "- How long is SIP?"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question about SIP...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_question, faq_database)

        completed_response = st.write_stream(
            type_response(response),
            cursor="▌",
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": completed_response,
        }
    )
