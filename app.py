import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="SIP Knowledge Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .source-box {
            background-color: #F4F8FC;
            border-left: 4px solid #1F4E78;
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin-top: 0.8rem;
        }

        .warning-box {
            background-color: #FFF8E1;
            border-left: 4px solid #C78A00;
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin-top: 0.8rem;
        }

        .no-answer-box {
            background-color: #FFF3F2;
            border-left: 4px solid #B42318;
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin-top: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_FOLDER = Path(__file__).resolve().parent
EXCEL_FILE = APP_FOLDER / "SIP_FAQ_Bot_Database.xlsx"

REQUIRED_COLUMNS = [
    "FAQ_ID",
    "Question",
    "Proposed_Answer",
    "Relevant_Policy_or_Source",
    "Source_Status",
    "Keywords",
]


# ==========================================================
# 2. TEXT-CLEANING FUNCTIONS
# ==========================================================

def normalise_text(text):
    """Convert text into a consistent form for comparison."""
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_keywords(value):
    """Turn a semicolon-separated Keywords cell into individual phrases."""
    value = "" if pd.isna(value) else str(value)
    return [
        normalise_text(part)
        for part in value.split(";")
        if normalise_text(part)
    ]


# ==========================================================
# 3. LOAD THE SIP FAQ DATABASE
# ==========================================================

@st.cache_data
def load_faq_database(path, modified_time):
    df = pd.read_excel(path, sheet_name="SIP FAQ Database")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Your Excel file is missing these columns: " + ", ".join(missing)
        )

    for column in REQUIRED_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    # If the workbook contains Record_Status, only use Published records.
    if "Record_Status" in df.columns:
        df["Record_Status"] = df["Record_Status"].fillna("").astype(str)
        published = df[
            df["Record_Status"].str.strip().str.lower().eq("published")
        ].copy()

        # For this POC, still load all records if none are currently Published.
        # This prevents a blank bot while the lecturer review status is pending.
        if not published.empty:
            df = published

    df["question_clean"] = df["Question"].apply(normalise_text)
    df["keywords_list"] = df["Keywords"].apply(split_keywords)

    # This is used only for similarity ranking AFTER safety/intent filtering.
    df["search_text"] = df.apply(
        lambda row: " ".join(
            [row["question_clean"]] + row["keywords_list"]
        ),
        axis=1,
    )

    return df.reset_index(drop=True)


def create_tfidf_index(df):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        min_df=1,
    )
    matrix = vectorizer.fit_transform(df["search_text"])
    return vectorizer, matrix


# ==========================================================
# 4. INTENT DEFINITIONS
# ==========================================================
# The order matters. Specific/high-risk issues are detected
# before generic workplace wording can influence the answer.

INTENTS = {
    "suspicious_email": {
        "triggers": [
            "suspicious email", "suspicious message", "phishing", "phish",
            "scam email", "scam message", "scam", "malicious email",
            "malicious link", "suspicious link", "fake email",
            "fraud email", "dodgy email", "strange email",
            "weird email", "unsafe link", "do not click",
        ],
        "faq_id": "SIP-029",
    },
    "injury_incident": {
        "triggers": [
            "injured", "injury", "accident", "hurt myself", "hurt at work",
            "workplace accident", "injured at work", "injured during sip",
            "injury report", "insurance claim", "medical incident",
        ],
        "faq_id": "SIP-024",
    },
    "mental_wellbeing": {
        "triggers": [
            "mental distress", "mental health", "severe stress",
            "emotionally distressed", "emotional distress", "counsellor",
            "counselor", "care tp", "care@tp", "wellbeing support",
            "need counselling", "need counseling",
        ],
        "faq_id": "SIP-010",
    },
    "mc_submission": {
        "triggers": [
            "medical certificate", "submit mc", "send mc", "my mc",
            "sick leave", "medical leave", "doctor certificate",
            "mc deadline", "unwell", "i am sick", "im sick",
        ],
        "faq_id": "SIP-001",
    },
    "mc_makeup": {
        "triggers": [
            "mc limit", "more than 5 days mc", "5 days mc",
            "make up days", "makeup days", "make-up days",
            "exceeded mc", "exceed mc", "8 days mc",
        ],
        "faq_id": "SIP-004",
    },
    "journal_word_limit": {
        "triggers": [
            "word limit", "word count", "journal length",
            "exceed word limit", "more words",
        ],
        "faq_id": "SIP-003",
    },
    "journal_late": {
        "triggers": [
            "forgot learning journal", "missed journal deadline",
            "late learning journal", "journal overdue",
            "submit journal late", "missed submission",
        ],
        "faq_id": "SIP-017",
    },
    "journal_schedule": {
        "triggers": [
            "learning journal schedule", "journal schedule",
            "learning journal deadline", "journal submission",
            "sip schedule", "overall schedule", "lms schedule",
            "submission dates",
        ],
        "faq_id": "SIP-002",
    },
    "overtime": {
        "triggers": [
            "overtime", "overtime work", "required to work late",
            "extra hours", "work extra hours", "ot",
        ],
        "faq_id": "SIP-018",
    },
    "work_from_home": {
        "triggers": [
            "work from home", "wfh", "remote work",
            "remote working", "work remotely",
        ],
        "faq_id": "SIP-020",
    },
    "document_signing": {
        "triggers": [
            "sign document", "sign a document", "workplace document",
            "contract", "indemnify", "indemnity", "non compete",
            "non-compete", "sign agreement", "do not understand document",
        ],
        "faq_id": "SIP-019",
    },
    "confidentiality": {
        "triggers": [
            "share workplace information", "confidential information",
            "confidentiality", "disclose information",
            "share company information", "share work information",
        ],
        "faq_id": "SIP-016",
    },
    "social_media": {
        "triggers": [
            "social media", "post internship", "post about work",
            "workplace photos", "take photos", "netiquette",
            "instagram", "linkedin",
        ],
        "faq_id": "SIP-021",
    },
    "personal_details": {
        "triggers": [
            "change personal information", "change contact details",
            "update contact details", "update personal details",
            "change phone number", "change my details",
        ],
        "faq_id": "SIP-022",
    },
    "inappropriate_behaviour": {
        "triggers": [
            "inappropriate behaviour", "inappropriate behavior",
            "misconduct", "harassment", "witness misconduct",
            "bad behaviour", "bad behavior",
        ],
        "faq_id": "SIP-023",
    },
    "property_damage": {
        "triggers": [
            "damage company property", "damaged property",
            "broke company property", "accidental damage",
            "damage equipment", "broke equipment",
        ],
        "faq_id": "SIP-026",
    },
    "not_enough_work": {
        "triggers": [
            "not enough work", "no work", "insufficient tasks",
            "idle at work", "nothing to do", "workload low",
        ],
        "faq_id": "SIP-025",
    },
    "mentorship": {
        "triggers": [
            "mentorship", "mentor", "mentoring",
            "who can mentor me", "guidance",
        ],
        "faq_id": "SIP-027",
    },
    "pay": {
        "triggers": [
            "increase pay", "salary increase", "unhappy pay",
            "pay increase", "ask hr for pay", "higher allowance",
            "pay negotiation", "raise",
        ],
        "faq_id": "SIP-028",
    },
    "dress_code": {
        "triggers": [
            "what to wear", "dress code", "work attire",
            "clothing", "clothes", "company shirt", "shorts",
            "slippers", "dress professionally",
        ],
        "faq_id": "SIP-009",
    },
    "assessment": {
        "triggers": [
            "performance assessment", "performance evaluation",
            "who assesses me", "who evaluates me", "sip assessment",
            "assessed during sip",
        ],
        "faq_id": "SIP-005",
    },
    "company_event": {
        "triggers": [
            "company event", "scholarship event", "attend event",
            "leave for event", "company programme",
        ],
        "faq_id": "SIP-006",
    },
    "report_presentation": {
        "triggers": [
            "final report", "sip report", "final presentation",
            "prepare report", "report preparation", "prepare early",
        ],
        "faq_id": "SIP-007",
    },
    "lateness": {
        "triggers": [
            "late for work", "late to work", "late to workplace",
            "punctuality", "on time", "commute", "travel time",
            "traffic", "inform supervisor late",
        ],
        "faq_id": "SIP-008",
    },
    "public_holiday": {
        "triggers": [
            "public holiday", "public holidays", "work on holiday",
            "attendance holiday", "sip holiday",
        ],
        "faq_id": "SIP-011",
    },
    "emergency_absence": {
        "triggers": [
            "unable to report", "cannot report to work",
            "emergency absence", "urgent situation",
            "absent emergency", "emergency",
        ],
        "faq_id": "SIP-012",
    },
    "untrained_task": {
        "triggers": [
            "not trained", "untrained task", "unfamiliar task",
            "task not trained", "new task",
        ],
        "faq_id": "SIP-013",
    },
    "supervisor_problem": {
        "triggers": [
            "problem with supervisor", "issue with supervisor",
            "conflict with supervisor", "complaint supervisor",
            "supervisor problem",
        ],
        "faq_id": "SIP-014",
    },
    "phone_use": {
        "triggers": [
            "personal phone", "mobile phone", "phone during work",
            "use phone", "phone working hours",
        ],
        "faq_id": "SIP-015",
    },
    "keep_notes": {
        "triggers": [
            "keep notes", "take notes", "work notes",
            "record work done", "internship notes",
            "notes for presentation", "document my work",
        ],
        "faq_id": "SIP-030",
    },
}


# ==========================================================
# 5. SAFE RETRIEVAL LOGIC
# ==========================================================

def find_intent_match(query_clean, faq_df):
    """
    Match clear topic phrases first.
    This prevents a specific question such as suspicious-email SOP
    from being confused with generic workplace FAQ entries.
    """
    for intent_name, config in INTENTS.items():
        if any(trigger in query_clean for trigger in config["triggers"]):
            result = faq_df[faq_df["FAQ_ID"].str.strip() == config["faq_id"]]
            if not result.empty:
                return result.iloc[0], intent_name
    return None, None


def fallback_similarity_match(query_clean, faq_df, vectorizer, matrix):
    """
    Used only when no specific intent is detected.
    Requires a strong score and clear word overlap; otherwise returns no answer.
    """
    query_words = {
        word for word in query_clean.split()
        if len(word) >= 4 and word not in {
            "what", "should", "would", "could", "during",
            "internship", "student", "please", "need",
        }
    }

    query_vector = vectorizer.transform([query_clean])
    similarity_scores = cosine_similarity(query_vector, matrix).flatten()

    candidates = []

    for index, row in faq_df.iterrows():
        row_words = {
            word
            for word in normalise_text(
                f"{row['Question']} {row['Keywords']}"
            ).split()
            if len(word) >= 4
        }

        overlap = query_words.intersection(row_words)
        similarity = float(similarity_scores[index])

        # A safe fallback requires at least:
        # - 2 meaningful common words, OR
        # - a strong similarity result.
        if len(overlap) >= 2 or similarity >= 0.42:
            candidates.append(
                {
                    "index": index,
                    "overlap_count": len(overlap),
                    "similarity": similarity,
                    "score": (len(overlap) * 0.25) + similarity,
                }
            )

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item["score"])

    # Final confidence gate.
    if best["overlap_count"] < 2 and best["similarity"] < 0.42:
        return None

    return faq_df.iloc[best["index"]]


def find_answer(query, faq_df, vectorizer, matrix):
    query_clean = normalise_text(query)

    if len(query_clean.split()) < 2:
        return None, "Please type a fuller SIP question."

    # First: exact FAQ wording.
    exact = faq_df[faq_df["question_clean"] == query_clean]
    if not exact.empty:
        return exact.iloc[0], "Exact FAQ match"

    # Second: high-precision intent matching.
    intent_row, intent_name = find_intent_match(query_clean, faq_df)
    if intent_row is not None:
        return intent_row, f"Topic match: {intent_name}"

    # Third: controlled similarity fallback for paraphrases.
    fallback_row = fallback_similarity_match(
        query_clean,
        faq_df,
        vectorizer,
        matrix,
    )

    if fallback_row is not None:
        return fallback_row, "High-confidence paraphrase match"

    # No weak/guessed result is returned.
    return None, "No sufficiently confident match found"


# ==========================================================
# 6. RESPONSE FORMATTING AND TYPEWRITER OUTPUT
# ==========================================================

def format_response(row):
    answer = row["Proposed_Answer"].strip()
    source = row["Relevant_Policy_or_Source"].strip()
    status = row["Source_Status"].strip()

    reply = (
        f"{answer}\n\n"
        f"**Source:** {source}\n\n"
        f"**Status:** {status}"
    )

    if status.lower() == "requires confirmation":
        reply += (
            "\n\n⚠️ **This item is marked Requires Confirmation. "
            "Please confirm it with your LO, lecturer, workplace supervisor, "
            "HR, or relevant staff before relying on it as final guidance.**"
        )

    return reply


def fast_typewriter(text):
    """
    Stream text word-by-word. Streamlit renders string chunks using a
    typewriter effect, and this speed stays readable without feeling slow.
    """
    chunks = re.split(r"(\s+)", text)

    for chunk in chunks:
        if chunk:
            yield chunk
            time.sleep(0.01)


# ==========================================================
# 7. STREAMLIT CHAT INTERFACE
# ==========================================================

st.title("🎓 SIP Knowledge Assistant")
st.caption(
    "Ask an SIP question in your own words. "
    "The chatbot returns only answers from the approved SIP FAQ database."
)

if not EXCEL_FILE.exists():
    st.error(
        "File not found: `SIP_FAQ_Bot_Database.xlsx`. "
        "Place it in the same folder as `app.py`."
    )
    st.stop()

try:
    faq_df = load_faq_database(
        str(EXCEL_FILE),
        EXCEL_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Unable to load the Excel FAQ database: {error}")
    st.stop()

if faq_df.empty:
    st.error("No FAQ records were loaded from the Excel file.")
    st.stop()

vectorizer, tfidf_matrix = create_tfidf_index(faq_df)

with st.sidebar:
    st.header("About this prototype")
    st.write(f"FAQ records loaded: **{len(faq_df)}**")
    st.caption(
        "The chatbot uses exact matches, specific-topic detection, and "
        "high-confidence paraphrase matching. It does not answer when "
        "there is insufficient evidence."
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi — I am the SIP Knowledge Assistant. Ask me about SIP in "
                "your own words, for example: “I received a suspicious email "
                "at work. What should I do?”"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

student_question = st.chat_input("Type your SIP question here...")

if student_question:
    st.session_state.messages.append(
        {"role": "user", "content": student_question}
    )

    with st.chat_message("user"):
        st.markdown(student_question)

    matched_row, match_type = find_answer(
        student_question,
        faq_df,
        vectorizer,
        tfidf_matrix,
    )

    with st.chat_message("assistant"):
        if matched_row is None:
            reply = (
                "I could not find a sufficiently confident answer in the current "
                "approved SIP FAQ.\n\n"
                "Please rephrase your question with more specific SIP details, "
                "or contact your LO or lecturer for clarification."
            )

            st.write_stream(fast_typewriter(reply))

            st.markdown(
                """
                <div class="no-answer-box">
                    <strong>Why I did not give an FAQ answer:</strong>
                    This prototype is configured to avoid guessing or returning
                    an unrelated answer when the match is weak.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            reply = format_response(matched_row)

            st.write_stream(fast_typewriter(reply))

            if matched_row["Source_Status"].strip().lower() == "requires confirmation":
                st.markdown(
                    """
                    <div class="warning-box">
                        <strong>Confirmation required:</strong>
                        Please verify this answer with the relevant LO, lecturer,
                        workplace supervisor, HR, or appropriate staff member.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )