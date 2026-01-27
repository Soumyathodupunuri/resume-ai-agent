import streamlit as st
import pdfplumber
import io
import requests
from bs4 import BeautifulSoup

from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load AI model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()



# Extract text from resume file
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

        document = Document(io.BytesIO(file))
        return "\n".join([p.text for p in document.paragraphs])


    # TXT
    else:
        return file.decode("utf-8")


# Fetch Job Description from Link
def fetch_job_description(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(response.text, "lxml")

        text = soup.get_text(separator=" ")

        return text.strip()

    except:
        return ""


# Clean important job info
def extract_keywords(text):

    sentences = text.split(".")
    important = sentences[:20]     # Top important lines

    return " ".join(important)


# Get AI embedding
def get_embedding(text):

    return model.encode(text)


# Skill list
SKILLS = [
    "python","java","sql","aws","docker","react",
    "node","flask","django","fastapi","ml","ai",
    "data analysis","linux","git","tensorflow",
    "pytorch","cloud","api","mongodb"
]
