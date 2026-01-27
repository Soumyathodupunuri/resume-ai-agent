import streamlit as st
import pdfplumber
import io
import requests
from bs4 import BeautifulSoup
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ----------------------------
# Load model with caching
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# Extract text from resume
# ----------------------------
def extract_text(file, name):
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file)) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        return text

    elif name.endswith(".docx"):
        document = Document(io.BytesIO(file))
        return "\n".join([p.text for p in document.paragraphs if p.text.strip()])

    else:
        return file.decode("utf-8")

# ----------------------------
# Fetch job description from URL
# ----------------------------
def fetch_job_description(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        # Remove scripts, styles
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        # Clean excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return ""

# ----------------------------
# Extract most important sentences
# ----------------------------
def extract_important_info(text, top_n=20):
    sentences = re.split(r'\.|\n', text)  # split by period or newline
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # remove empty/short
    # prioritize sentences with keywords
    important_keywords = ["skills", "experience", "qualification", "requirement", "responsibility", "job"]
    important = [s for s in sentences if any(k in s.lower() for k in important_keywords)]
    return " ".join(important[:top_n])  # top N important sentences

# ----------------------------
# Skill matching
# ----------------------------
SKILLS = [
    "python","java","sql","aws","docker","react","node","flask","django","fastapi",
    "ml","ai","data analysis","linux","git","tensorflow","pytorch","cloud","api","mongodb"
]

def match_skills(text):
    text_lower = text.lower()
    matched = [skill for skill in SKILLS if skill in text_lower]
    return matched

# ----------------------------
# Similarity score
# ----------------------------
def similarity_score(resume_text, job_text):
    emb_resume = model.encode([resume_text])
    emb_job = model.encode([job_text])
    return cosine_similarity(emb_resume, emb_job)[0][0]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Smart Resume AI Agent")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
job_url = st.text_input("Paste Job Link Here:")

if uploaded_file and job_url:
    resume_text = extract_text(uploaded_file.read(), uploaded_file.name)
    job_text_raw = fetch_job_description(job_url)
    job_text = extract_important_info(job_text_raw)

    st.subheader("Important Job Info Extracted:")
    st.write(job_text if job_text else "Could not extract meaningful info.")

    st.subheader("Matched Skills")
    st.write(match_skills(resume_text))

    st.subheader("Resume vs Job Similarity")
    score = similarity_score(resume_text, job_text)
    st.write(f"{score:.2f} (0=low match, 1=perfect match)")
