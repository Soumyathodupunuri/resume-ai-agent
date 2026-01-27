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
def generate_creative_resume(name, role, skills):

    profile = (
        f"Dynamic and result-oriented {role} with strong expertise in "
        f"{', '.join(skills[:4])}. Skilled in building scalable systems, "
        f"optimizing performance, and delivering high-impact solutions."
    )

    resume = f"""

{name.upper()}
{role}
India | 📞 Phone | ✉ Email | 🔗 LinkedIn


PROFILE OVERVIEW
{profile}


TECHNICAL EXPERTISE
{' | '.join(skills)}


PROFESSIONAL CONTRIBUTIONS
AI Resume Optimization Platform
• Designed NLP-powered ATS scoring system using transformers
• Improved resume matching accuracy by 45%
• Built cloud-ready Streamlit application

Job Portal Management System
• Engineered secure backend using Flask and JWT authentication
• Reduced API response time by 35%
• Integrated role-based user management


KEY PROJECTS & INNOVATIONS
Smart Career Recommendation Engine
• Developed ML-based ranking algorithms
• Integrated real-time job scraping APIs
• Increased placement success rate by 28%


TOOLS, FRAMEWORKS & PLATFORMS
{' | '.join(skills[:7])}


ACADEMIC CREDENTIALS
B.Tech in Computer Science – XYZ University (2023–2027)


DISTINCTIONS & CERTIFICATIONS
• Google Cloud Foundations
• AWS Cloud Practitioner (Ongoing)
• LeetCode Top 15%

"""

    return resume




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
st.markdown("Built by Soumya | Resume AI Agent")
