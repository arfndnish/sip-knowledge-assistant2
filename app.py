import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="SIP FAQ Chatbot",
    page_icon="🎓",
    layout="centered",
)

DATABASE_FILE = "SIP_FAQ_Bot_Database_Complete.xlsx"
MATCH_THRESHOLD = 0.16


@st.cache_data
def load_faq_database(file_name):
    file_path = Path(file_name)

    if not file_path.exists():
        return None

    dataframe = pd.read_excel(file_path)
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

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

    dataframe = dataframe.fillna("")

    dataframe = dataframe[
        dataframe["Record_Status"].astype(str).str.lower() == "published"
    ].copy()

    dataframe["Search_Text"] = (
        dataframe["Question"].astype(str)
        + " "
        + dataframe["Keywords"].astype(str).str.replace(";", " ", regex=False)
    )

    return dataframe


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_best_faq(user_question, faq_database):
    user_question = clean_text(user_question)

    searchable_faqs = faq_database["Search_Text"].apply(clean_text).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(searchable_faqs + [user_question])

    similarity_scores = cosine_similarity(
        vectors[-1],
        vectors[:-1],
    ).flatten()

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]

    return faq_database.iloc[best_index], best_score


faq_database = load_faq_database(DATABASE_FILE)

st.title("🎓 SIP FAQ Chatbot")
st.write(
    "Ask a question about your Student Internship Programme (SIP). "
    "This chatbot searches the approved SIP FAQ database and supports "
    "different ways of phrasing the same question."
)

if faq_database is None:
    st.error(
        f"Database not found: `{DATABASE_FILE}`. "
        "Upload the Excel file to the same GitHub folder as `app.py`."
    )
    st.stop()

with st.expander("Try asking questions such as"):
    st.write(
        "- How do I submit my MC?\n"
        "- Can I work on public holidays?\n"
        "- What do I wear for my internship?\n"
        "- I might be late for work—what should I do?\n"
        "- How long is SIP?\n"
        "- I received a suspicious email at work."
    )

user_question = st.text_input(
    "Type your SIP question:",
    placeholder="Example: I am sick. Where do I send my medical certificate?",
)

if st.button("Get answer", type="primary"):
    if not user_question.strip():
        st.warning("Please type a SIP-related question first.")
    else:
        best_faq, confidence_score = find_best_faq(
            user_question,
            faq_database,
        )

        if confidence_score < MATCH_THRESHOLD:
            st.warning(
                "I could not find a confident answer in the current SIP FAQ database."
            )
            st.write(
                "Please check SIP Teams or contact your Learning Officer (LO) "
                "for clarification."
            )
        else:
            st.subheader("Answer")
            st.write(best_faq["Proposed_Answer"])

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.caption(f"FAQ ID: {best_faq['FAQ_ID']}")
                st.caption(
                    f"Source status: {best_faq['Source_Status']}"
                )

            with col2:
                st.caption(
                    f"Source: {best_faq['Relevant_Policy_or_Source']}"
                )

            with st.expander("Matched FAQ record"):
                st.write(best_faq["Question"])

            with st.expander("Keywords recognised by this record"):
                st.write(best_faq["Keywords"])

            with st.expander("Match confidence"):
                st.write(
                    f"Similarity score: {confidence_score:.2f}. "
                    "A higher score means the question more closely matches "
                    "the FAQ wording or keyword phrases."
                )
