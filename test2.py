from langchain_community.llms import HuggingFaceEndpoint

# from getpass import getpass
from dotenv import load_dotenv

# HUGGINGFACEHUB_API_TOKEN = getpass()

import os

load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# question = "Who won the FIFA World Cup in the year 1994? "

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate.from_template(template)



# llm_chain = LLMChain(prompt=prompt, llm=llm)
# print(llm_chain.run(question))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_pinecone import PineconeVectorStore


import streamlit as st

# Retrieve Pinecone API key from environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pincone embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
index_name = "nlp"

vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)


# path = "try2\Modi-Ki-Guarantee-Sankalp-Patra-English_2.pdf"

def load_document(path):

    # Load PDF document
    loader = PyPDFLoader(path)
    data = loader.load()
    # print(data)

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(data)
    # len(text_chunks)

    # Initialize Pinecone vector store
    vector_store_from_docs = PineconeVectorStore.from_documents(text_chunks, index_name=index_name, embedding=embeddings)


# Add documents to Pinecone index
# vector_store.add_documents(text_chunks)

# since pinecone is being used for vector store
# vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)



#  not in use so far
# ---------------------------------------------------------------
# system_prompt = (
#     "Use the given context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentence maximum and keep the answer concise. "
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# ----------------------------------------------------------------

# query = "How has the Indian government empowered farmers through various initiatives, including MSP hikes, procurement, and income support programs?"

# result_similar = vector_store.similarity_search(query)
# print('SEARCH KA RESULTTTTTTTTTTTTTTT')
# print(result_similar)

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

def ask_query(query):

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_store.as_retriever())
    result = qa.invoke(query)
    print(result)
    
    # Access and display the value of the 'result' key
    st.write(result['result'])


# Add a function to trigger the resume summarization prompt automatically
def summarize_resume():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )
    # This will ask for the summarization of the resume
    query = "Summarize the resume of the candidate."
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = qa.invoke(query)
    print(result)
    
    # Display the result on the Streamlit interface
    st.write(result['result'])

# ask_query(query)

# Function to generate job description
def generate_job_description(job_role):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)

    job_description_prompt = f"Generate a job description for a {job_role}. Include the job title, required skills, and responsibilities."
    result = llm(job_description_prompt)
    st.caption("Detailed Job Decription")
    st.write(result)


# Function to generate Q&A based on candidate name
def generate_qa(candidate_name, global_job_role):
    # Initialize model and embeddings
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)

    # Search for candidate in vector store
    query = f"Retrieve information about the candidate named {candidate_name}."
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = qa.invoke(query)

    # If candidate not found, show message
    if not result:
        st.write(f"No resume found for candidate {candidate_name}. Please make sure the candidate exists in the database.")
        return
    
    st.write(f"Resume found for {candidate_name}. Generating questions based on the resume and job description...")

    # Generate 10 questions dynamically based on the candidate's resume and job description
    question_prompt = f"Based on the resume and skills of {candidate_name} and {global_job_role}, generate a set of 10 interview questions."
    
    questions_result = llm(question_prompt)
    
    # Display the questions
    st.write("Generated Questions:")
    st.write(questions_result)


# Create the "uploads" directory if it doesn't exist
uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)



# Create sidebar for navigation
st.sidebar.title("HR Process AI Automation")
tab = st.sidebar.radio("Select a Tab", ["CV Analyzer", "Job Description Generator", "Generate QA", "LDA analysis"])

global_job_role=''

# Tab 1: CV Analyzer
if tab == "CV Analyzer":
    st.title("CV Analyzer")

    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # Generate a unique filename (optional)
        filename = f"{os.path.splitext(uploaded_file.name)[0]}.pdf"  # Remove extension and add .pdf
        unique_filename = filename
        counter = 1
        while os.path.exists(os.path.join(uploads_dir, unique_filename)):
            unique_filename = f"{filename[:-4]}_{counter}.pdf"  # Add counter before extension
            counter += 1

        # Save the uploaded file in the "uploads" directory
        with open(os.path.join(uploads_dir, unique_filename), "wb") as f:
            f.write(uploaded_file.read())
        path = os.path.join(uploads_dir, unique_filename)  # Update path with full directory
        load_document(path)

        # Automatically summarize the resume after the document is processed
        st.success(f"Document uploaded and processed! (Saved as: {unique_filename})")

        # Call the resume summarization function
        summarize_resume()


    user_query = st.text_input("Enter your question:")

    if st.button("Ask") and user_query and uploaded_file:
        # response = 
        st.caption("Result:")
        ask_query(user_query)
        # st.write(response)

# Tab 2: Job Description Generator
elif tab == "Job Description Generator":
    st.title("Job Description Generator")
    
    job_role = st.text_input("Enter the job role (e.g., Python Developer, Cloud Engineer):")
    global_job_role=job_role
    if st.button("Generate Job Description") and job_role:
        generate_job_description(job_role)

# Tab 3: Generate Q nad A
elif tab == "Generate QA":
    st.title("Generate QA")
    
    candidate = st.text_input("Enter the name:")
    
    if st.button("Enter Name") and candidate:
        generate_qa(candidate, global_job_role)