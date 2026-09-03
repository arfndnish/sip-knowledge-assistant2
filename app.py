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
    initial_sidebar_state="collapsed",
)

DATABASE_FILE = "SIP_FAQ_Bot_Database_Complete.xlsx"
SIMILARITY_THRESHOLD = 0.12


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_data
def load_faq_database(file_name):
    file_path = Path(file_name)

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

    shared_words = user_words.intersection(keyword_words)
    return len(shared_words) / len(user_words)


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
        lambda keywords: keyword_overlap_score(cleaned_question, keywords)
    ).to_numpy()

    combined_scores = (similarity_scores * 0.75) + (keyword_scores * 0.25)

    best_index = combined_scores.argmax()
    best_score = combined_scores[best_index]

    return faq_database.iloc[best_index], best_score


def format_answer(faq_record):
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

    return format_answer(faq_record)


def response_stream(text):
    words = text.split(" ")

    for word in words:
        yield word + " "
        time.sleep(0.018)


def clear_chat():
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
    "Your AI-style guide for frequently asked Student Internship Programme questions."
)

if faq_database is None:
    st.error(
        f"I cannot find `{DATABASE_FILE}`. "
        "Please upload it to the same GitHub folder as `app.py`."
    )
    st.stop()

if "messages" not in st.session_state:
    clear_chat()

with st.sidebar:
    st.subheader("Chat controls")

    if st.button("🗑️ Clear chat"):
        clear_chat()
        st.rerun()

    st.caption(f"Knowledge base: {len(faq_database)} published SIP FAQ records")

    with st.expander("Example questions"):
        st.write(
            "- I am sick. Where should I send my MC?\n"
            "- Can I work from home during SIP?\n"
            "- I might be late because of traffic.\n"
            "- What if I get injured at work?\n"
            "- How long is SIP?"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input(
    "Ask a question about SIP...",
)

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
            answer = generate_response(user_question, faq_database)

        streamed_answer = st.write_stream(
            response_stream(answer),
            cursor="▌",
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": streamed_answer,
        }
    )
