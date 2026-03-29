import streamlit as st
from chat import generate_answer

st.set_page_config(
    page_title="Chatbot Admitere UniTBv",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Chatbot Admitere UniTBv")
st.write("Pune întrebări despre medii de admitere, taxe, acte sau confirmarea locului.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bună! Sunt asistentul virtual pentru admiterea la UniTBv. Cu ce te pot ajuta?"
        }
    ]

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

with st.sidebar:
    st.header("Opțiuni")
    st.session_state.show_debug = st.checkbox("Afișează fragmentele găsite", value=False)

    if st.button("Șterge conversația"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bună! Sunt asistentul virtual pentru admiterea la UniTBv. Cu ce te pot ajuta?"
            }
        ]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Scrie întrebarea ta aici...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Se caută răspunsul..."):
            raspuns, docs_gasite = generate_answer(user_input)

        st.markdown(raspuns)

        if st.session_state.show_debug:
            with st.expander("Fragmente folosite pentru răspuns"):
                for i, doc in enumerate(docs_gasite, start=1):
                    sursa = doc.metadata.get("sursa", "necunoscută")
                    program = doc.metadata.get("program", "")
                    st.markdown(f"**Fragment {i}** | sursa: `{sursa}` | program: `{program}`")
                    st.code(doc.page_content, language="text")

    st.session_state.messages.append({"role": "assistant", "content": raspuns})