# Custom HKILA PPE Study Bot

# Required libraries
import os
import openai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# ======================= CONFIGURATION =======================

# --- (1) Set your OpenAI API key safely ---
# Recommended: store in your environment with: export OPENAI_API_KEY="your-key"
openai.api_key = os.getenv("OPENAI_API_KEY")  # <-- Set this in your system or below for testing only
# openai.api_key = "sk-xxxx"  # <-- Uncomment for temporary direct use (not recommended for public code)

# --- (2) Google Drive Service Account JSON (already uploaded)
SERVICE_ACCOUNT_FILE = "gpt-access-468102-f29a5df6c2c9.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1yTiGfVpSlTRFmqJgJfXogM92HQA5wNfw'

# ======================= AUTHENTICATE GOOGLE DRIVE =======================

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# ======================= LIST & DOWNLOAD PDFs =======================

def list_pdfs_in_drive_folder(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)").execute()
    return results.get('files', [])

def download_pdf(file_id, file_name):
    request = drive_service.files().get_media(fileId=file_id)
    with open(file_name, 'wb') as f:
        f.write(request.execute())

# ======================= EXTRACT TEXT FROM PDF =======================

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or '' for page in reader.pages)

# ======================= CREATE VECTOR INDEX =======================

def create_vectorstore_from_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ======================= CREATE QA CHAIN =======================

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), retriever=retriever)

# ======================= MAIN FUNCTION =======================

def main():
    print("Downloading and indexing documents from Google Drive folder...")
    pdfs = list_pdfs_in_drive_folder(FOLDER_ID)
    full_text = ""

    for pdf in pdfs:
        print(f"Downloading: {pdf['name']}")
        download_pdf(pdf['id'], pdf['name'])
        full_text += extract_text_from_pdf(pdf['name']) + "\n"

    print("Creating searchable knowledge base...")
    vectorstore = create_vectorstore_from_text(full_text)
    qa_chain = create_qa_chain(vectorstore)

    print("\nReady! Type your HKILA PPE question below (or 'exit' to quit):\n")
    while True:
        query = input("Q: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print(f"A: {answer}\n")

if __name__ == "__main__":
    main()
