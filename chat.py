from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_admitere",embedding_function=embeddings)

llm=OllamaLLM(model="qwen2.5:7b", temperature=0)

system_prompt = (
    "### IDENTITATE ȘI ROL ###\n"
    "Ești asistentul virtual OFICIAL al Universității Transilvania din Brașov (UniTBv). "
    "Misiunea ta este unică și strictă: asistență pentru ADMITERE.\n\n"

    "### REGULI DE CITIRE A TABELULUI ###\n"
    "Datele sunt structurate astfel: Program | Nota max bug | Nota min bug | Nota max taxă | Nota min taxă.\n"
    "1. Când utilizatorul întreabă de 'ultima medie la buget', extrage STRICT a DOUA cifră de după numele programului.\n"
    "2. Identifică programul cu atenție maximă la detalii (ex: 'Autovehicule rutiere' este diferit de 'Autovehicule rutiere în limba engleză').\n"
    "3. Dacă găsești programul solicitat, răspunde formatat: 'La programul [Nume], ultima medie la buget a fost [Cifra]'.\n\n"
    
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
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 15}),question_answer_chain)



print("\n---- Chatbot Admitere Unitbv este Gata! (Scrie exit pentru a iesi) ---")

while True:
    user_input = input("Tu:")
    if user_input.lower() == 'exit':
        break

    retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    docs_gasite = retriever.invoke(user_input)

    print("\n--- DEBUG: CONTEXT EXTRAS DIN BAZA DE DATE ---")
    for i, doc in enumerate(docs_gasite):
        print(f"Fragment {i+1}:\n{doc.page_content}\n")
    print("-----------------------------------------------\n")

    raspuns = rag_chain.invoke({"input":user_input})
    print(f"\nChatbot:{raspuns['answer']}\n")