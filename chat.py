from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_admitere",embedding_function=embeddings)

llm=OllamaLLM(model="qwen2.5:7b")

system_prompt = (
    "Ești un asistent strict și specializat DOAR pe procesul de admitere la Universitatea Transilvania din Brașov. "
    "Sarcina ta este să răspunzi la întrebări folosind EXCLUSIV informațiile din contextul furnizat mai jos. "
    "\n\n"
    "REGULI CRITICE:\n"
    "1. Dacă întrebarea utilizatorului NU are legătură cu admiterea, facultățile, taxele, actele UniTBv, mediile de admitere "
    "răspunde exact așa: 'Ne pare rău, dar pot oferi informații doar despre procesul de admitere la UniTBv.'\n"
    "2. NU folosi cunoștințele tale generale pentru a răspunde la subiecte precum geografie, istorie, matematică generală sau alte domenii.\n"
    "3. Răspunde întotdeauna în limba română.\n"
    "4. Dacă informația lipsește din contextul de mai jos, dar întrebarea este despre admitere, îndrumă-i către admitere.unitbv.ro.\n"
    "\n\n"
    "CONTEXT DISPONIBIL:\n"
    "{context}"
    
)

prompt = ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    ('human',"{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(),question_answer_chain)



print("\n---- Chatbot Admitere Unitbv este Gata! (Scrie exit pentru a iesi) ---")

while True:
    user_input = input("Tu:")
    if user_input.lower() == 'exit':
        break

    raspuns = rag_chain.invoke({"input":user_input})
    print(f"\nChatbot:{raspuns['answer']}\n")