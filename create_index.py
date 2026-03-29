import json
import os
import re
import unicodedata
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"{field_name}:\s*(.*?)\s*(\||$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def create_database():
    if os.path.exists("./db_admitere"):
        import shutil
        shutil.rmtree("./db_admitere")
        print("Baza de date veche a fost ștearsă.")

    documents = []

    if os.path.exists("date_admitere.json"):
        with open("date_admitere.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            complete_text = f"Întrebare: {item['question']}\nRăspuns: {item['answer']}"
            doc = Document(
                page_content=complete_text,
                metadata={
                    "sursa": "faq_data",
                    "tip_info": "administrativ"
                }
            )
            documents.append(doc)

        print(f"Am încărcat {len(data)} întrebări din JSON.")


    md_path = "source_docs/date_curate2.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            facultate = extract_field(line, "FACULTATE")
            domeniu = extract_field(line, "DOMENIU")
            program = extract_field(line, "PROGRAM")

            doc = Document(
                page_content=line,
                metadata={
                    "sursa": "tabel_medii",
                    "tip_info": "note_admitere",
                    "facultate": facultate,
                    "domeniu": domeniu,
                    "program": program,
                    "facultate_norm": normalize_text(facultate),
                    "domeniu_norm": normalize_text(domeniu),
                    "program_norm": normalize_text(program),
                }
            )
            documents.append(doc)

        print(f"Am încărcat {len(lines)} fragmente din Markdown.")

 
    scraped_path = "scraped_content/unitbv_pages.json"
    if os.path.exists(scraped_path):
        with open(scraped_path, "r", encoding="utf-8") as f:
            scraped_pages = json.load(f)

        count_docs = 0

        for item in scraped_pages:
            facultate = item.get("facultate", "")
            section_name = item.get("section_name", "")
            tip_info = item.get("tip_info", "")
            section_url = item.get("section_url", "")
            content = item.get("content", "")

            if not content.strip():
                continue

            full_text = (
                f"FACULTATE: {facultate}\n"
                f"SECȚIUNE: {section_name}\n"
                f"TIP_INFO: {tip_info}\n"
                f"URL: {section_url}\n\n"
                f"{content}"
            )

            doc = Document(
                page_content=full_text,
                metadata={
                    "sursa": "site_unitbv",
                    "tip_info": tip_info,
                    "facultate": facultate,
                    "sectiune": section_name,
                    "url": section_url,
                    "facultate_norm": normalize_text(facultate),
                    "sectiune_norm": normalize_text(section_name),
                }
            )

            documents.append(doc)
            count_docs += 1

        print(f"Am încărcat {count_docs} documente din paginile scrape-uite.")

    if not documents:
        print("Eroare: Nu am găsit nicio sursă de date!")
        return

    print(f"Total documente de indexat: {len(documents)}")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./db_admitere"
    )

    print("Succes! Baza de date a fost creată.")


if __name__ == "__main__":
    create_database()