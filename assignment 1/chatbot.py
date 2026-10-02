import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Streamlit Page
# -----------------------------
st.title("💬  Welcome to my Chatbot")
st.write("Chatbot using Streamlit, Ollama and LangChain")


# -----------------------------
# Ollama Model
# -----------------------------
llm = OllamaLLM(model="llama3.2")


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful chatbot.
    Answer the user's question clearly and simply.

    User Question: {question}

    Answer:
    """
)


# -----------------------------
# LLM Chain
# -----------------------------
chain = prompt | llm


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Clear Chat Button
# -----------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# -----------------------------
# Show Previous Messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# User Question
# -----------------------------
question = st.chat_input("Ask something...")


if question:

    # Show user question
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    answer = chain.invoke({
        "question": question
    })

    # Show chatbot answer
    with st.chat_message("assistant"):
        st.write(answer)

    # Save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Save chatbot answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })