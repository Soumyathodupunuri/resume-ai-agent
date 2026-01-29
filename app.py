import streamlit as st
import pdfplumber
import io
import requests
from bs4 import BeautifulSoup

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline


# ---------------- LOAD MODELS ----------------

model = SentenceTransformer("all-MiniLM-L6-v2")

ai_generator = pipeline(
    "text-generation",
    model="gpt2"
)


# ---------------- FUNCTIONS ----------------

# Extract resume text
def extract_text(file, name):

    if name.endswith(".pdf"):

        text = ""

        with pdfplumber.open(io.BytesIO(file)) as pdf:

            for page in pdf.pages:

                if page.extract_text():
                    text += page.extract_text()

        return text


    elif name.endswith(".docx"):

        doc = Document(io.BytesIO(file))
        return "\n".join([p.text for p in doc.paragraphs])


    else:
        return file.decode("utf-8")


# Get job description from URL
def get_jd_from_link(url):

    res = requests.get(url)

    soup = BeautifulSoup(res.text, "html.parser")

    text = soup.get_text(separator=" ")

    return text[:5000]   # limit text size


# Embeddings
def get_embedding(text):

    return model.encode(text)


# Skill list
SKILLS = [
    "python","java","sql","aws","docker","react",
    "node","flask","django","fastapi","ml","ai",
    "data analysis","linux","git","cloud",
    "kubernetes","pandas","numpy"
]


# Extract skills
def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        if skill in text:
            found.append(skill)

    return found


# ATS Score
def calculate_ats(resume, jd):

    resume_words = set(resume.lower().split())
    jd_words = set(jd.lower().split())

    matched = resume_words & jd_words

    score = (len(matched) / len(jd_words)) * 100

    return round(score, 2)


# AI Resume Optimizer
def optimize_resume(resume, jd):

    prompt = f"""
Rewrite this resume to match the job description.

Add missing keywords.
Make it ATS friendly.
Professional tone.

Resume:
{resume}

Job Description:
{jd}
"""

    result = ai_generator(
        prompt,
        max_length=700,
        do_sample=False
    )

    return result[0]["generated_text"]


# ---------------- UI ----------------

st.set_page_config(page_title="AI Resume Agent")

st.title("🤖 AI Resume Matching System")

st.write("Upload Resume + Paste Job Link")


# Upload resume
resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf","docx","txt"]
)


# Job URL
job_url = st.text_input(
    "Paste Job Link"
)


# Analyze
if st.button("Analyze"):


    if resume_file is None or job_url == "":

        st.error("Upload Resume and Paste Job Link")

    else:


        with st.spinner("AI Processing..."):


            # Resume
            resume_bytes = resume_file.read()

            resume_text = extract_text(
                resume_bytes,
                resume_file.name
            )


            # Job Description
            jd_text = get_jd_from_link(job_url)


            # Embeddings
            resume_vec = get_embedding(resume_text)
            jd_vec = get_embedding(jd_text)


            # Match score
            match_score = cosine_similarity(
                [resume_vec],
                [jd_vec]
            )[0][0] * 100


            # ATS score
            ats_score = calculate_ats(resume_text, jd_text)


            # Skills
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            matched = set(resume_skills) & set(jd_skills)
            missing = set(jd_skills) - set(resume_skills)


            # AI optimization
            improved_resume = optimize_resume(
                resume_text,
                jd_text
            )


        # ---------------- RESULTS ----------------

        st.success("Analysis Complete")


        st.subheader("📊 Scores")

        st.write(f"Match Score: {round(match_score,2)}%")
        st.write(f"ATS Score: {ats_score}%")


        st.subheader("✅ Matched Skills")
        st.write(list(matched))


        st.subheader("❌ Missing Skills")
        st.write(list(missing))


        st.subheader("📝 AI Optimized Resume")

        st.text_area(
            "Improved Resume",
            improved_resume,
            height=400
        )
