import re
import unicodedata
from typing import List, Optional

from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text



embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_db = Chroma(
    persist_directory="./db_admitere",
    embedding_function=embeddings
)


llm = OllamaLLM(model="qwen2.5:7b", temperature=0)

all_docs = vector_db.get()
documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

vector_retriever = vector_db.as_retriever(search_kwargs={"k": 15})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.85, 0.15]
)

compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)


system_prompt = (
    "### IDENTITATE ȘI ROL ###\n"
    "Ești asistentul virtual oficial al Universității Transilvania din Brașov (UniTBv). "
    "Oferi informații exclusiv despre admiterea la UniTBv.\n\n"

    "### TIPURI DE ÎNTREBĂRI PERMISE ###\n"
    "- medii de admitere\n"
    "- programe de studii\n"
    "- condiții de admitere\n"
    "- calendarul admiterii\n"
    "- număr de locuri\n"
    "- taxe\n"
    "- acte necesare pentru înscriere\n"
    "- confirmarea locului\n"
    "- informații suplimentare despre admitere\n"
    "- contact\n\n"

    "### REGULI PENTRU MEDII ###\n"
    "Fragmentele din sursa 'tabel_medii' au formatul:\n"
    "FACULTATE | DOMENIU | PROGRAM | ... | MEDII: MaxBug | MinBug | MaxTaxa | MinTaxa.\n"
    "Interpretarea corectă este:\n"
    "- Nota maximă la buget = prima valoare după MEDII\n"
    "- Nota minimă la buget = a doua valoare după MEDII\n"
    "- Nota maximă la taxă = a treia valoare după MEDII\n"
    "- Nota minimă la taxă = a patra valoare după MEDII\n\n"

    "### REGULI SUPLIMENTARE PENTRU ÎNTREBĂRI DESPRE MEDII ###\n"
    "- Dacă utilizatorul întreabă la plural despre 'mediile de admitere' sau 'notele de admitere', oferă toate valorile disponibile:\n"
    "  maxim buget, minim buget, maxim taxă, minim taxă.\n"
    "- Dacă există mai multe rezultate relevante pentru același termen, specifică programul sau domeniul pentru fiecare rezultat.\n"
    "- Dacă există un match exact pe PROGRAM, prioritizează acel rezultat față de match-ul pe DOMENIU.\n"
    "- Nu spune că nu ai informația dacă ea există clar în tabelul din context.\n\n"

    "### REGULI DE RĂSPUNS ###\n"
    "- Răspunde exclusiv în limba română.\n"
    "- Folosește doar informația din context.\n"
    "- Nu inventa date.\n"
    "- Dacă informația nu există clar în context, spune că nu o deții și trimite utilizatorul către admitere.unitbv.ro.\n"
    "- Dacă utilizatorul cere altceva decât informații despre admiterea la UniTBv, răspunde strict:\n"
    "'Ne pare rău, dar pot oferi informații doar despre procesul de admitere la UniTBv.'\n\n"

    "### CONTEXT ###\n"
    "{context}\n\n"

    "### INSTRUCȚIUNI ###\n"
    "- Dacă întrebarea este despre medii, identifică exact programul, domeniul sau facultatea corectă.\n"
    "- Dacă întrebarea este despre taxe, acte, calendar, programe sau confirmarea locului, răspunde din secțiunea relevantă.\n"
    "- Dacă există mai multe fragmente relevante, folosește-le pe cele mai importante.\n"
    "- Răspunde scurt, clar și la obiect."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)



def detect_question_type(user_input: str) -> Optional[str]:
    query = normalize_text(user_input)

   
    if any(word in query for word in [
        "cum se plateste",
        "cum platesc",
        "cum se achita",
        "cum achit",
        "plata taxei",
        "achitare",
        "taxa de studiu",
        "taxa de scolarizare",
        "prima transa",
        "transfer bancar",
        "cum platesc taxa",
        "cum se plateste taxa"
    ]):
        return "taxe"

  
    if any(word in query for word in [
        "nota minima",
        "nota maxima",
        "ultima medie",
        "medie",
        "medii",
        "mediile de admitere",
        "medii de admitere",
        "notele de admitere",
        "note de admitere",
        "note admitere",
        "la buget",
        "la taxa",
        "admitere la taxa",
        "admitere la buget"
    ]):
        return "note_admitere"

    if any(word in query for word in [
        "acte", "documente", "dosar", "ce trebuie sa incarc",
        "ce trebuie sa depun", "ce documente"
    ]):
        return "acte_inscriere"

    if any(word in query for word in [
        "calendar", "termen", "cand", "afisarea listelor",
        "pana cand", "cand se afiseaza"
    ]):
        return "calendar_admitere"

    if any(word in query for word in [
        "taxa de inscriere", "taxe", "cost", "cat costa", "cat este taxa"
    ]):
        return "taxe"

    if any(word in query for word in [
        "programe", "specializari", "ce programe", "ce specializari",
        "program de studiu", "programe de studii"
    ]):
        return "programe_studiu"

    if any(word in query for word in [
        "confirm", "confirmarea locului", "cum confirm locul",
        "cum se confirma locul"
    ]):
        return "confirmare_loc"

    if any(word in query for word in [
        "contact", "telefon", "email", "e-mail", "adresa"
    ]):
        return "contact"

    return None


def score_note_candidates(user_input: str, docs: List[Document]) -> List[Document]:
    query_norm = normalize_text(user_input)
    query_words = set(query_norm.split())

    scored = []

    for doc in docs:
        if doc.metadata.get("sursa") != "tabel_medii":
            continue

        score = 0

        program_norm = doc.metadata.get("program_norm", "")
        domeniu_norm = doc.metadata.get("domeniu_norm", "")
        facultate_norm = doc.metadata.get("facultate_norm", "")
        alias_program_norm = doc.metadata.get("alias_program_norm", "")

        
        if program_norm and program_norm in query_norm:
            score += 100

        if alias_program_norm and alias_program_norm in query_norm:
            score += 80

        if domeniu_norm and domeniu_norm in query_norm:
            score += 50
       
        if facultate_norm and facultate_norm in query_norm:
            score += 30


        for field_value, bonus in [
            (program_norm, 20),
            (domeniu_norm, 10),
            (facultate_norm, 5),
        ]:
            if field_value:
                field_words = set(field_value.split())
                overlap = len(query_words.intersection(field_words))
                score += overlap * bonus

        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored]


def filter_docs_by_tip(tip_info: str, docs: List[Document]) -> List[Document]:
    return [doc for doc in docs if doc.metadata.get("tip_info") == tip_info]


def get_local_bm25_results(user_input: str, local_docs: List[Document], k: int = 5) -> List[Document]:
    if not local_docs:
        return []

    local_bm25 = BM25Retriever.from_documents(local_docs)
    local_bm25.k = min(k, len(local_docs))
    return local_bm25.invoke(user_input)


def smart_retrieve(user_input: str) -> List[Document]:
    question_type = detect_question_type(user_input)

    if question_type == "note_admitere":
        scored_matches = score_note_candidates(user_input, documents)

        if scored_matches:
            query_norm = normalize_text(user_input)
            exact_program_docs = []

            for doc in scored_matches:
                program_norm = doc.metadata.get("program_norm", "")
                alias_program_norm = doc.metadata.get("alias_program_norm", "")

                if program_norm and program_norm in query_norm:
                    exact_program_docs.append(doc)
                elif alias_program_norm and alias_program_norm in query_norm:
                    exact_program_docs.append(doc)

            if exact_program_docs:
                return exact_program_docs[:5]

            return scored_matches[:5]


    if question_type:
        filtered_docs = filter_docs_by_tip(question_type, documents)
        if filtered_docs:
            return get_local_bm25_results(user_input, filtered_docs, k=5)


    return compression_retriever.invoke(user_input)



def generate_answer(user_input: str):
    docs_gasite = smart_retrieve(user_input)

    raspuns = question_answer_chain.invoke({
        "input": user_input,
        "context": docs_gasite
    })

    return raspuns, docs_gasite