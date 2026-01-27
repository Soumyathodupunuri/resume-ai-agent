# ===============================
# ATS-Optimized Resume Analyzer
# ===============================
import streamlit as st
import pdfplumber
import io
import requests
from bs4 import BeautifulSoup
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ----------------------------
# Load AI model
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
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"Failed to fetch job description: {e}")
        return ""

# ----------------------------
# Extract most important sentences
# ----------------------------
def extract_important_info(text, top_n=20):
    sentences = re.split(r'\.|\n', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    # Prioritize sentences with keywords
    important_keywords = ["skills", "experience", "qualification", "requirement", "responsibility", "job"]
    important = [s for s in sentences if any(k in s.lower() for k in important_keywords)]
    
    if len(important) < top_n:
        remaining = [s for s in sentences if s not in important]
        important += remaining[:top_n - len(important)]

    return " ".join(important[:top_n])

# ----------------------------
# Skill list
# ----------------------------
SKILLS = [
    "python","java","sql","aws","docker","react","node","flask","django","fastapi",
    "ml","ai","data analysis","linux","git","tensorflow","pytorch","cloud","api","mongodb"
]

# ----------------------------
# Skill matching
# ----------------------------
def match_skills(text):
    text_lower = text.lower()
    matched = [skill for skill in SKILLS if skill in text_lower]
    return matched

def unmatched_skills(resume_text, job_text):
    matched = match_skills(resume_text)
    job_keywords = match_skills(job_text)
    missing = [skill for skill in job_keywords if skill not in matched]
    return missing

# ----------------------------
# ATS score
# ----------------------------
def ats_score(resume_text, job_text):
    matched = match_skills(resume_text)
    job_keywords = match_skills(job_text)
    if not job_keywords:
        return 0
    score = len(matched) / len(job_keywords)
    return round(score * 100, 2)

# ----------------------------
# Suggest companies based on skills
# ----------------------------
def suggest_companies(matched_skills):
    suggested = []
    if "python" in matched_skills and "ml" in matched_skills:
        suggested += ["Google", "Microsoft", "Amazon"]
    if "react" in matched_skills and "node" in matched_skills:
        suggested += ["Facebook", "Shopify", "Tesla"]
    if "aws" in matched_skills or "cloud" in matched_skills:
        suggested += ["IBM", "Oracle", "Accenture"]
    return list(set(suggested))  # remove duplicates

# ----------------------------
# Revise resume and create PDF
# ----------------------------
def create_revised_pdf(resume_text, missing_skills):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph("Revised Resume", styles['Title']))
    flowables.append(Spacer(1,12))
    
    # Add original resume text
    flowables.append(Paragraph("Original Resume Content:", styles['Heading2']))
    for line in resume_text.split("\n"):
        if line.strip():
            flowables.append(Paragraph(line, styles['Normal']))
    
    flowables.append(Spacer(1,12))
    
    # Add missing skills section
    if missing_skills:
        flowables.append(Paragraph("Added Skills to Improve ATS Score:", styles['Heading2']))
        flowables.append(Paragraph(", ".join(missing_skills), styles['Normal']))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("ATS-Optimized Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
job_url = st.text_input("Paste Job Link Here:")

if st.button("Analyze & Revise"):
    if not uploaded_file:
        st.warning("Please upload a resume file first!")
    elif not job_url:
        st.warning("Please paste a job link first!")
    else:
        resume_text = extract_text(uploaded_file.read(), uploaded_file.name)
        job_text_raw = fetch_job_description(job_url)
        job_text = extract_important_info(job_text_raw)

        # Matched / unmatched skills
        matched = match_skills(resume_text)
        missing = unmatched_skills(resume_text, job_text)

        st.subheader("Matched Skills")
        st.write(matched if matched else "No matched skills found.")

        st.subheader("Unmatched Skills (to add for 100% ATS)")
        st.write(missing if missing else "All skills matched!")

        # ATS score
        score = ats_score(resume_text, job_text)
        st.subheader("ATS Score")
        st.write(f"{score}%")

        # Suggested companies
        companies = suggest_companies(matched)
        st.subheader("Suggested Companies Based on Skills")
        st.write(companies if companies else "No specific company suggestions.")

        # Generate revised PDF
        revised_pdf = create_revised_pdf(resume_text, missing)
        st.download_button(
            label="Download Revised Resume PDF",
            data=revised_pdf,
            file_name="Revised_Resume.pdf",
            mime="application/pdf"
        )
