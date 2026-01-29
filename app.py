import streamlit as st
import pdfplumber
import io
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import tempfile

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Smart ATS Resume Builder", layout="centered")
st.title("📄 Smart ATS-Friendly Resume Builder")
st.write("Upload your old resume, paste the job description, and generate an optimized resume with matched skills highlighted.")

# -------------------------------
# Read Resume
# -------------------------------
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
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file.decode("utf-8", errors="ignore")

# -------------------------------
# Skills Extraction
# -------------------------------
TECH_KEYWORDS = [
    "python","java","c","c++","sql","aws","ml","ai","flask","django",
    "react","node","git","docker","kubernetes","linux","tensorflow",
    "pandas","numpy","matlab","verilog","vhdl","arduino","raspberry pi",
    "stm32","iot","fpga","asic","eda tools","microcontrollers"
]

def extract_skills(text):
    text = text.lower()
    skills_found = []
    for skill in TECH_KEYWORDS:
        if skill in text:
            skills_found.append(skill)
    return list(set(skills_found))

# -------------------------------
# Build Resume
# -------------------------------
def build_resume(old_text, jd_text, name, contact):
    old_skills = extract_skills(old_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = sorted(list(set(old_skills) & set(jd_skills)))
    unmatched_skills = sorted(list(set(jd_skills) - set(old_skills)))

    # Sections
    sections = {
        "SUMMARY": [],
        "SKILLS": [],
        "PROJECTS": [],
        "EXPERIENCE": [],
        "EDUCATION": []
    }

    lines = [l.strip() for l in old_text.split("\n") if l.strip()]
    for line in lines:
        l = line.lower()
        if "project" in l:
            sections["PROJECTS"].append(line)
        elif "intern" in l or "experience" in l:
            sections["EXPERIENCE"].append(line)
        elif "b.tech" in l or "degree" in l or "university" in l:
            sections["EDUCATION"].append(line)
        elif any(word in l for word in TECH_KEYWORDS):
            sections["SKILLS"].append(line)
        else:
            sections["SUMMARY"].append(line)

    # ---------------- Build Resume Text ----------------
    resume = f"{name}\n{contact}\n\n"

    # Summary
    resume += "SUMMARY\n" + "-"*40 + "\n"
    summary_text = "Motivated engineering student with strong problem-solving skills and experience in building applications."
    if matched_skills:
        summary_text += " Skilled in " + ", ".join(matched_skills) + "."
    resume += summary_text + "\n\n"

    # Skills
    resume += "SKILLS\n" + "-"*40 + "\n"
    if matched_skills:
        resume += "✅ Matched Skills: " + ", ".join(matched_skills) + "\n"
    if unmatched_skills:
        resume += "⚠️ Skills to Learn: " + ", ".join(unmatched_skills) + "\n"
    resume += "\n"

    # Projects
    resume += "PROJECTS\n" + "-"*40 + "\n"
    for p in sections["PROJECTS"][:5]:
        resume += f"- {p}\n"
    resume += "\n"

    # Experience
    resume += "EXPERIENCE\n" + "-"*40 + "\n"
    for e in sections["EXPERIENCE"][:4]:
        resume += f"- {e}\n"
    resume += "\n"

    # Education
    resume += "EDUCATION\n" + "-"*40 + "\n"
    for ed in sections["EDUCATION"][:3]:
        resume += f"- {ed}\n"

    return resume, matched_skills, unmatched_skills

# -------------------------------
# PDF Generator
# -------------------------------
def generate_pdf(text):
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    path = file.name

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "heading", parent=styles["Normal"],
        fontSize=13, spaceAfter=8, spaceBefore=12,
        alignment=TA_LEFT, bold=True
    )
    normal = styles["Normal"]
    story = []

    for line in text.split("\n"):
        if not line.strip():
            story.append(Spacer(1,10))
            continue
        if line.isupper() and len(line) < 25:
            p = Paragraph(f"<b>{line}</b>", heading)
        else:
            p = Paragraph(line, normal)
        story.append(p)
        story.append(Spacer(1,6))

    doc.build(story)
    return path

# -------------------------------
# Streamlit UI
# -------------------------------
# Use session state to store input reliably
if "name" not in st.session_state:
    st.session_state.name = ""
if "contact" not in st.session_state:
    st.session_state.contact = ""

st.subheader("👤 Your Details")
st.text_input("Full Name", key="name")
st.text_input("Email | Phone | LinkedIn | GitHub", key="contact")

st.subheader("📎 Upload Resume")
resume_file = st.file_uploader("Upload Old Resume (PDF / DOCX / TXT)", ["pdf", "docx", "txt"])

st.subheader("💼 Job Information")
job_link = st.text_input("Job Link (Optional)")
job_desc = st.text_area("Paste Job Description Here", height=200)

# Generate Resume
if st.button("Generate ATS Resume with Skills"):
    name = st.session_state.name.strip()
    contact = st.session_state.contact.strip()

    if not resume_file:
        st.error("❌ Please upload your resume")
        st.stop()
    if not name or not contact:
        st.error("❌ Enter your name and contact details")
        st.stop()
    if not job_desc.strip():
        st.error("❌ Paste job description")
        st.stop()

    with st.spinner("Building optimized resume..."):
        old_text = extract_text(resume_file.read(), resume_file.name)
        new_resume, matched, unmatched = build_resume(old_text, job_desc, name, contact)
        pdf_path = generate_pdf(new_resume)

    st.success("✅ Resume Generated Successfully")

    # Display matched/unmatched skills
    st.subheader("📊 Skill Analysis")
    if matched:
        st.write("✅ Matched Skills: " + ", ".join(matched))
    if unmatched:
        st.write("⚠️ Skills to Learn: " + ", ".join(unmatched))

    if job_link.strip():
        st.info(f"🔗 Job Link: {job_link}")

    # Download PDF
    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download ATS Resume (PDF)",
            f,
            file_name="ATS_Optimized_Resume.pdf",
            mime="application/pdf"
        )
