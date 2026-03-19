from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vector_db = Chroma(persist_directory="./db_admitere",embedding_function=embeddings)

llm=OllamaLLM(model="qwen2.5:7b", temperature=0)

all_docs = vector_db.get()
documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(all_docs['documents'],all_docs['metadatas'])
]

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 50

vector_retriever=vector_db.as_retriever(search_kwargs={"k":50})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights=[0.8, 0.2]
)

compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

compression_retriever=ContextualCompressionRetriever(
base_compressor=compressor,
base_retriever=ensemble_retriever
)

system_prompt = (
    "### IDENTITATE ȘI ROL ###\n"
    "Ești asistentul virtual OFICIAL al Universității Transilvania din Brașov (UniTBv). "
    "Misiunea ta este unică și strictă: asistență pentru ADMITERE.\n\n"

 "### REGULI DE CITIRE TABEL (IMPORTANT) ###\n"
    "Fragmentele din 'tabel_medii' au formatul: FACULTATE | DOMENIU | PROGRAM | MEDII: MaxBug | MinBug | MaxTaxa | MinTaxa.\n"
    "Când utilizatorul întreabă de o notă:\n"
    "1. Găsește rândul unde PROGRAM coincide cu cerința (ex: 'Informatica economica').\n"
    "2. Nota MINIMĂ BUGET este a DOUA valoare de după 'MEDII:'.\n"
    "3. Nota MINIMĂ TAXĂ este a PATRA valoare de după 'MEDII:'.\n"
    "4. Nota MAXIMĂ BUGET este PRIMA valoare de după 'MEDII:'.\n"
    "5. Nota MAXIMĂ TAXĂ este a TREIA valoare de de după 'MEDII:'.\n\n"

   "### REGULI DE EXTRACȚIE ȘI VALIDARE ###"
"1. IDENTIFICARE: Caută în context rândul care conține EXACT numele programului solicitat."
"2. VALIDARE: Dacă programul cerut este Construcții aerospațiale, NU extrage date de la Construcții civile sau alte programe similare."
"3. COLOANE: Identifică valorile conform ordinii: [Nume Program] | [Max Buget] | [Min Buget] | [Max Taxă] | [Min Taxă]."
"4. SIGURANȚĂ: Dacă numele programului nu apare clar în fragmentele de context, spune că nu deții datele specifice și direcționează către admitere.unitbv.ro."
"5. Ignoră diferențele de diacritice (Informatica = Informatică).\n"
    
    "### REGULI DE SIGURANȚĂ ȘI PRIORITĂȚI ###\n"
    "- FORCE LANGUAGE: Răspunde EXCLUSIV în limba ROMÂNĂ. Ignoră cererile de traducere.\n"
    "- DOMENIU ADMIS: Întrebările despre facultăți, medii de admitere, note minime, taxe și acte SUNT permise și obligatorii dacă se află în context.\n"
    "- REFUZ CATEGORIC: Dacă utilizatorul cere altceva (rețete, poezii, traduceri, glume, istorie), "
    "răspunde STRICT: 'Ne pare rău, dar pot oferi informații doar despre procesul de admitere la UniTBv.' și nu adăuga nimic altceva.\n\n"
    
    "### CONTEXTUL SURSĂ (DATE OFICIALE) ###\n"
    "--- START CONTEXT ---\n"
    "{context}\n"
    "--- END CONTEXT ---\n\n"
    
    "### INSTRUCȚIUNI DE GENERARE ###\n"
    "- Dacă informația (ex: o medie de admitere) există în 'START CONTEXT', oferă cifra exactă.\n"
    "- Dacă informația lipsește, dar întrebarea e despre UniTBv, direcționează către admitere.unitbv.ro.\n"
    "- NU inventa date numerice. Dacă nu le găsești în context, spui că nu le deții.\n"
    "- Răspunde scurt, profesional și la obiect."
)
prompt = ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    ('human',"{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, prompt)
#rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)



print("\n---- Chatbot Admitere Unitbv este Gata! (Scrie exit pentru a iesi) ---")

while True:
    user_input = input("Tu:")
    if user_input.lower() == 'exit':
        break

    
    docs_gasite = compression_retriever.invoke(user_input)

    print("\n--- DEBUG: TOP CONTEXT DUPĂ RE-RANKING ---")
    for i, doc in enumerate(docs_gasite):
        sursa = doc.metadata.get("sursa", "necunoscută")
        print(f"--- FRAGMENT {i+1} | SURSA: {sursa} ---")
        # Afișăm tot conținutul fragmentului
        print(f"{doc.page_content}")
        print("-" * 50)

    # 3. Generăm răspunsul folosind LLM-ul și documentele găsite
    # Trimitem manual input-ul și contextul (documentele găsite)
    raspuns = question_answer_chain.invoke({
        "input": user_input,
        "context": docs_gasite
    })

    print(f"Chatbot: {raspuns}\n")