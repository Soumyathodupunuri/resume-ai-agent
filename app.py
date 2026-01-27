import streamlit as st
import pdfplumber
import io
import requests
import re

from bs4 import BeautifulSoup
from docx import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ---------------------------------
# Load AI Model
# ---------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ---------------------------------
# Extract Resume Text
# ---------------------------------

def extract_text(file, name):

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file)) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text

    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(file))
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        return file.decode("utf-8")


# ---------------------------------
# Fetch Job Description
# ---------------------------------

def fetch_job_description(url):

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)

        soup = BeautifulSoup(r.text, "lxml")

        text = soup.get_text(" ")

        return re.sub("\s+", " ", text)

    except:
        return ""


# ---------------------------------
# Skill Database
# ---------------------------------

SKILLS = [
    "python","java","c++","sql","aws","docker","react","node","flask",
    "django","fastapi","ml","ai","nlp","tensorflow","pytorch","git",
    "linux","mongodb","mysql","cloud","api","data analysis"
]


# ---------------------------------
# Extract Skills
# ---------------------------------

def extract_skills(text):

    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return list(set(found))


# ---------------------------------
# ATS Score
# ---------------------------------

def calculate_ats(resume, job):

    emb1 = model.encode([resume])
    emb2 = model.encode([job])

    score = cosine_similarity(emb1, emb2)[0][0]

    return round(score * 100, 2)


# ---------------------------------
# Improve Resume
# ---------------------------------

def improve_resume(resume, job):

    job_words = set(job.lower().split())
    resume_words = set(resume.lower().split())

    missing = list(job_words - resume_words)
    keywords = missing[:12]

    improved = []

    improved.append("SUMMARY\n")
    improved.append(
        "Motivated Computer Science student with expertise in "
        + ", ".join(keywords[:5]) +
        ". Strong problem-solving and software development skills."
    )

    improved.append("\n\nTECHNICAL SKILLS\n")
    improved.append(
        "Programming & Tools: " +
        ", ".join(sorted(set(keywords + extract_skills(resume))))
    )

    improved.append("\n\nPROJECTS\n")

    improved.append(
        "• Developed AI-powered resume analyzer using Python and NLP, "
        "improving ATS compatibility by 35%."
    )

    improved.append(
        "• Built full-stack job portal using Flask and SQL with secure authentication."
    )

    improved.append("\n\nEXPERIENCE\n")

    improved.append(
        "• Automated data pipelines using Python, reducing processing time by 40%."
    )

    improved.append("\n\nACHIEVEMENTS\n")

    improved.append(
        "• Ranked among top performers in coding competitions and hackathons."
    )

    return "\n".join(improved)


# ---------------------------------
# Create PDF
# ---------------------------------

def create_pdf(text, filename="optimized_resume.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    elements = []

    for line in text.split("\n\n"):

        if line.isupper() or line.endswith("\n"):
            elements.append(Paragraph(f"<b>{line}</b>", styles["Heading2"]))
        else:
            elements.append(Paragraph(line.replace("\n","<br/>"), styles["Normal"]))

        elements.append(Spacer(1,15))

    doc.build(elements)

    return filename


# ---------------------------------
# Streamlit UI
# ---------------------------------

st.set_page_config(page_title="Resume AI Agent", layout="wide")

st.title("📄 Resume AI Analyzer & Optimizer")

st.write("Upload Resume + Paste Job Link → Get ATS Score + Optimized Resume")


# Upload
resume_file = st.file_uploader(
    "Upload Resume (PDF/DOCX/TXT)",
    type=["pdf","docx","txt"]
)

# Job Link
job_url = st.text_input("Paste Job Description Link")


if resume_file and job_url:

    if st.button("Analyze Resume"):

        with st.spinner("Processing..."):

            # Resume Text
            resume_text = extract_text(
                resume_file.read(),
                resume_file.name.lower()
            )

            # Job Text
            job_text = fetch_job_description(job_url)

            if job_text == "":
                st.error("Could not fetch job description.")
                st.stop()

            # Skills
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            matched = list(set(resume_skills) & set(job_skills))
            missing = list(set(job_skills) - set(resume_skills))

            # ATS
            ats = calculate_ats(resume_text, job_text)

            # Improve Resume
            improved = improve_resume(resume_text, job_text)

            # Create PDF
            pdf_file = create_pdf(improved)


        # ---------------- Results ----------------

        st.success("Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched Skills")
            st.write(matched)

            st.subheader("❌ Missing Skills")
            st.write(missing)

        with col2:
            st.subheader("📊 ATS Score")
            st.metric("Match %", f"{ats}%")

        st.subheader("📝 Optimized Resume")

        st.text(improved)


        # Download PDF
        with open(pdf_file, "rb") as f:

            st.download_button(
                "⬇️ Download Resume PDF",
                f,
                file_name="optimized_resume.pdf"
            )


# Footer
st.markdown("---")
st.markdown("Built by Soumya | Resume AI Agent")")
