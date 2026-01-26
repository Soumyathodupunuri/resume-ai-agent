import streamlit as st
import pdfplumber
import docx
import io

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from file
def extract_text(file, name):

    # PDF
    if name.endswith(".pdf"):

        text = ""
        with pdfplumber.open(io.BytesIO(file)) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()

        return text


    # DOCX
    elif name.endswith(".docx"):

        document = docx.Document(io.BytesIO(file))
        return "\n".join([p.text for p in document.paragraphs])


    # TXT
    else:
        return file.decode("utf-8")


# Get AI embedding
def get_embedding(text):

    return model.encode(text)


# Skill list
SKILLS = [
    "python","java","sql","aws","docker","react",
    "node","flask","django","fastapi","ml","ai",
    "data analysis","linux","git"
]


# Extract skills
def extract_skills(text):

    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return found


# UI
st.set_page_config(page_title="AI Resume Matcher")

st.title("🤖 AI Resume Matching Agent")

st.write("Upload Resume and Job Description")


# Upload files
resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf","docx","txt"]
)

jd_file = st.file_uploader(
    "Upload Job Description",
    type=["pdf","docx","txt"]
)


# Button
if st.button("Analyze"):

    if resume_file is None or jd_file is None:

        st.error("Please upload both files")

    else:

        with st.spinner("AI is analyzing..."):

            # Read files
            resume_bytes = resume_file.read()
            jd_bytes = jd_file.read()

            # Extract text
            resume_text = extract_text(
                resume_bytes,
                resume_file.name
            )

            jd_text = extract_text(
                jd_bytes,
                jd_file.name
            )

            # Get embeddings
            resume_vec = get_embedding(resume_text)
            jd_vec = get_embedding(jd_text)

            # Similarity
            score = cosine_similarity(
                [resume_vec],
                [jd_vec]
            )[0][0] * 100


            # Skills
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            matched = set(resume_skills) & set(jd_skills)
            missing = set(jd_skills) - set(resume_skills)


        # Results
        st.success(f"Match Score: {round(score,2)}%")

        st.subheader("Matched Skills")
        st.write(list(matched))

        st.subheader("Missing Skills")
        st.write(list(missing))
