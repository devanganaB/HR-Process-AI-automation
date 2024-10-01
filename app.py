import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_pinecone import PineconeVectorStore


load_dotenv()

# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_API_TOKEN= "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"

# Hugging Face LLM
def get_llm():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    return HuggingFaceEndpoint(repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)

llm = get_llm()

# Function to generate job description
def generate_job_description(job_title):
    prompt = f"Generate a detailed job description for the position of {job_title}, including responsibilities and requirements in 150 words."
    # result = llm(prompt)
    result = llm.invoke(prompt)
    return result

# Function to analyze CV and summarize it in 200 words
def analyze_cv(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(data)
    
    text_to_summarize = ' '.join([chunk.page_content for chunk in text_chunks[:2]])  # Summarize first few chunks
    prompt = f"Summarize the following CV content in 200 words:\n{text_to_summarize}"
    
    summary = llm(prompt)
    return summary

# Function to generate screening questions
def generate_questions(job_desc, cv_summary, candidate_name):
    prompt = f"Based on the following job description: {job_desc}, and CV summary: {cv_summary}, generate 10 interview questions for a candidate named {candidate_name}."
    questions = llm(prompt)
    return questions

# Streamlit UI
st.title("HR Process AI Automation")

# Job Description Generator
st.header("Job Description Generator")
job_title = st.text_input("Enter Job Title (e.g., Python Developer):")

if st.button("Generate Job Description"):
    if job_title:
        try:
            job_desc = generate_job_description(job_title)
            st.write("### Generated Job Description:")
            st.write(job_desc)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a Job Title.")

# CV Analyzer
st.header("CV Analyzer")
uploaded_file = st.file_uploader("Upload a CV (PDF format):", type=["pdf"])
if uploaded_file is not None:
    with open("uploaded_cv.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("Analyze CV"):
        cv_summary = analyze_cv("uploaded_cv.pdf")
        st.write("### CV Summary:")
        st.write(cv_summary)

# Screening Question Generator
st.header("Generate Screening Questions")
candidate_name = st.text_input("Enter Candidate's Name:")
if st.button("Generate Questions"):
    if not job_title or not uploaded_file:
        st.warning("Please generate Job Description and upload CV first.")
    else:
        cv_summary = analyze_cv("uploaded_cv.pdf")
        questions = generate_questions(job_desc, cv_summary, candidate_name)
        st.write("### Screening Questions:")
        st.write(questions)
