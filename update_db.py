from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Calea către PDF-ul tău
cale_pdf = "source_docs\Medii_admitere_2024.pdf" 

print("Se încarcă PDF-ul...")
loader = PyPDFLoader(cale_pdf)
pagini = loader.load()

# 2. Împărțim textul (pentru ca AI-ul să găsească exact tabelul/media de care are nevoie)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
bucati_text = text_splitter.split_documents(pagini)

# 3. Modelul de embeddings (TREBUIE să fie același ca în chat.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Accesăm baza de date existentă și ADĂUGĂM noile informații
print("Se adaugă datele în baza de date...")
vector_db = Chroma(persist_directory="./db_admitere", embedding_function=embeddings)
vector_db.add_documents(bucati_text)

print("Succes! Mediile de admitere 2024 au fost adăugate în baza de date.")