import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_database():
    # Șterge folderul vechi pentru a asigura o indexare curată
    if os.path.exists("./db_admitere"):
        import shutil
        shutil.rmtree("./db_admitere")
        print("Baza de date veche a fost ștearsă.")

    documents = [] 

    # 1. Incarcarea datelor din FAQ de pe site-ul universitatii
    if os.path.exists("date_admitere.json"):
        with open("date_admitere.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            complete_text = f"Întrebare: {item['question']}\nRăspuns: {item['answer']}"
            doc = Document(page_content=complete_text, metadata={"sursa": "faq_data"})
            documents.append(doc)
        print(f"Am încărcat {len(data)} întrebări din JSON.")

    # 2. Incărcare datelor despre programele de studii si medii
    md_path = "source_docs/date_curate.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Splitter optimizat pentru tabele Markdown
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n"]
        )
        
        md_chunks = text_splitter.split_text(md_content)
        for chunk in md_chunks:
            doc = Document(page_content=chunk, metadata={"sursa": "tabel_medii"})
            documents.append(doc)
        print(f"Am încărcat {len(md_chunks)} fragmente din fișierul Markdown.")

    if not documents:
        print("Eroare: Nu am găsit nicio sursă de date!")
        return

    # 3. Generare Embeddings și Salvare
    print("Se descarcă modelul de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Se creează baza de date în folderul 'db_admitere'...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./db_admitere"
    )

    print("Succes! Baza de date a fost creată cu datele curățate.")

if __name__ == "__main__":
    create_database()