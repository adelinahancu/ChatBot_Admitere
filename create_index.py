import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def create_database():

    if not os.path.exists("date_admitere.json"):
        print("Eroare:Nu am gasit fisierul date_admitere.json!")
        return
    
    with open("date_admitere.json","r",encoding='utf-8') as f:
        data = json.load(f)

    documents = [] 
    for item in data:
        complete_text = f"Intrebare:{item['question']}\nRaspuns:{item['answer']}"
        doc = Document(page_content=complete_text, metadata={"sursa":"faq_data"})
        documents.append(doc)

    print("Downloanding embedding model(just for the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating the vector database in folder 'db_admitere' ...")
    vector_db=Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./db_admitere"
    )

    print("Success! The database was created.")

if __name__ == "__main__":
    create_database()