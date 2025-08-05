
import streamlit as st
import os
import openai
import tempfile
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# ========== CONFIG ==========
openai.api_key = st.secrets["openai"]["api_key"]
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_ID = st.secrets["google_drive"]["folder_id"]

# ========== GOOGLE DRIVE AUTH ==========
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    json.dump(dict(st.secrets["gcp_service_account"]), tmp)
    SERVICE_ACCOUNT_FILE = tmp.name

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

# ========== FUNCTIONS ==========
def list_pdfs_recursively(folder_id):
    files = []

    def _recurse(current_folder_id):
        response = drive_service.files().list(
            q=f"'{current_folder_id}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        for file in response.get("files", []):
            if file["mimeType"] == "application/pdf":
                files.append(file)
            elif file["mimeType"] == "application/vnd.google-apps.folder":
                _recurse(file["id"])

    _recurse(folder_id)
    return files

def download_pdf(file_id, file_name):
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_name, "wb") as f:
        f.write(request.execute())

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def create_vectorstore_from_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="HKILA PPE Study Bot", layout="wide")
st.title("📘 HKILA PPE Study Bot")

if st.button("📂 Load & Index PDFs from Google Drive"):
    with st.spinner("Processing all PDFs recursively..."):
        pdfs = list_pdfs_recursively(FOLDER_ID)
        full_text = ""
        for pdf in pdfs:
            download_pdf(pdf["id"], pdf["name"])
            full_text += extract_text_from_pdf(pdf["name"]) + "\n"
        vectorstore = create_vectorstore_from_text(full_text)
        st.session_state.qa_chain = create_qa_chain(vectorstore)
        st.success(f"Indexed {len(pdfs)} PDF(s) successfully!")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the documents:")
    if user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(user_question)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Please load the PDFs first by clicking the button above.")
