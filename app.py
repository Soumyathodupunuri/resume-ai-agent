import streamlit as st
import pdfplumber
import io
import requests

from bs4 import BeautifulSoup
from docx import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    text_ai = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    return embed_model, text_ai


model, ai_generator = load_models()


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

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    res = requests.get(url, headers=headers)

    soup = BeautifulSoup(res.text, "html.parser")

    text = soup.get_text(separator=" ")

    return text[:6000]


# Get embedding
def get_embedding(text):

    return model.encode(text)


# Skill database
SKILLS = [
    "python","java","sql","aws","docker","react",
    "node","flask","django","fastapi","ml","ai",
    "data analysis","linux","git","cloud",
    "kubernetes","pandas","numpy","spark",
    "tensorflow","pytorch"
]


# Extract skills
def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        if skill in text:
            found.append(skill)

    return found


# ATS score
def calculate_ats(resume, jd):

    resume_words = set(resume.lower().split())

    jd_words = set(jd.lower().split())

    matched = resume_words & jd_words

    if len(jd_words) == 0:
        return 0

    score = (len(matched) / len(jd_words)) * 100

    return round(score, 2)


# AI Resume optimizer
def optimize_resume(resume, jd):

    prompt = f"""
Rewrite this resume to strongly match the job description.

Make it:
- ATS friendly
- Keyword optimized
- Professional
- Well structured

Resume:
{resume}

Job Description:
{jd}
"""

    result = ai_generator(
        prompt,
        max_new_tokens=250,
        do_sample=False
    )

    return result[0]["generated_text"]


# Create PDF
def generate_pdf(text):

    buffer = io.BytesIO()

    c = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4

    x = 40
    y = height - 50


    for line in text.split("\n"):

        if y < 50:

            c.showPage()
            y = height - 50


        c.drawString(x, y, line)

        y -= 15


    c.save()

    buffer.seek(0)

    return buffer


# ---------------- UI ----------------

st.set_page_config(
    page_title="AI Resume Agent",
    layout="centered"
)


st.title("🤖 AI Resume Matching & ATS Optimizer")

st.write("Upload Resume + Paste Job Link")


# Upload Resume
resume_file = st.file_uploader(
    "📄 Upload Resume",
    type=["pdf","docx","txt"]
)


# Job Link
job_url = st.text_input(
    "🔗 Paste Job Link"
)


# Analyze Button
if st.button("🚀 Analyze Resume"):


    if resume_file is None or job_url.strip() == "":

        st.error("Please upload resume and paste job link")

    else:


        with st.spinner("AI Processing... Please wait"):


            # Resume text
            resume_bytes = resume_file.read()

            resume_text = extract_text(
                resume_bytes,
                resume_file.name
            )


            # Job description
            jd_text = get_jd_from_link(job_url)


            # Embeddings
            resume_vec = get_embedding(resume_text)

            jd_vec = get_embedding(jd_text)


            # Match Score
            match_score = cosine_similarity(
                [resume_vec],
                [jd_vec]
            )[0][0] * 100


            # ATS Score
            ats_score = calculate_ats(
                resume_text,
                jd_text
            )


            # Skills
            resume_skills = extract_skills(resume_text)

            jd_skills = extract_skills(jd_text)

            matched = set(resume_skills) & set(jd_skills)

            missing = set(jd_skills) - set(resume_skills)


            # AI Resume
            improved_resume = optimize_resume(
                resume_text,
                jd_text
            )


        # ---------------- RESULTS ----------------


        st.success("✅ Analysis Completed")


        st.subheader("📊 Resume Scores")

        st.metric("Match Score", f"{round(match_score,2)} %")

        st.metric("ATS Score", f"{ats_score} %")


        st.subheader("✅ Matched Skills")

        st.write(list(matched))


        st.subheader("❌ Missing Skills")

        st.write(list(missing))


        st.subheader("📝 AI Optimized Resume")

        st.text_area(
            "Improved Resume",
            improved_resume,
            height=350
        )


        # Download PDF
        pdf_file = generate_pdf(improved_resume)


        st.download_button(
            label="⬇️ Download Resume PDF",
            data=pdf_file,
            file_name="AI_Optimized_Resume.pdf",
            mime="application/pdf"
        )
