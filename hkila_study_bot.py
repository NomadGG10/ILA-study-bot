
# Custom HKILA PPE Study Bot with Recursive PDF Loading

import os
import openai
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import json

# ========== SECRETS SETUP ==========

# Save service account to temporary file from Streamlit secrets
with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
    json.dump(dict(st.secrets["gcp_service_account"]), f)
    temp_service_account_path = f.name

# Authenticate Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_file(
    temp_service_account_path, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

FOLDER_ID = st.secrets["google_drive"]["folder_id"]
openai.api_key = st.secrets["openai"]["api_key"]

# ========== PDF UTILITIES ==========

def list_pdfs_in_drive_folder_recursive(folder_id):
    pdfs = []

    def traverse_folder(fid):
        results = drive_service.files().list(
            q=f"'{fid}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.google-apps.folder')",
            fields="files(id, name, mimeType)").execute()
        for file in results.get('files', []):
            if file['mimeType'] == 'application/pdf':
                pdfs.append({'id': file['id'], 'name': file['name']})
            elif file['mimeType'] == 'application/vnd.google-apps.folder':
                traverse_folder(file['id'])

    traverse_folder(folder_id)
    return pdfs

def download_pdf(file_id, file_name):
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_name, 'wb') as f:
        f.write(request.execute())

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or '' for page in reader.pages)

# ========== LANGCHAIN SETUP ==========

def create_vectorstore_from_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key=openai.api_key), retriever=retriever)

# ========== STREAMLIT APP ==========

def main():
    st.set_page_config(page_title="HKILA PPE Study Bot", layout="wide")
    st.title("📘 HKILA PPE Study Bot")
    st.markdown("Ask any question about your uploaded PPE study documents from Google Drive.")

    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading and indexing documents from Google Drive..."):
            pdfs = list_pdfs_in_drive_folder_recursive(FOLDER_ID)
            full_text = ""
            for pdf in pdfs:
                download_pdf(pdf['id'], pdf['name'])
                full_text += extract_text_from_pdf(pdf['name']) + "\n"
            vectorstore = create_vectorstore_from_text(full_text)
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.success("Documents indexed successfully!")

    user_question = st.text_input("Enter your question below:")
    if user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(user_question)
            st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
