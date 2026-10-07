import streamlit as st
import ollama

# Page title
st.title("Context + User Query")

st.write("Enter a context and ask a question.")
context = st.text_area(
    "Enter Context",
    height=150,
    placeholder="Example: Python is a programming language used for AI and data science."
)


question = st.text_input(
    "Enter Your Question",
    placeholder="Example: What is Python used for?"
)

# Button
if st.button("Get Answer"):

    if context == "" or question == "":
        st.warning("Please enter both context and question.")

    else:
        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer the question using the context.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        st.subheader("AI Response")
        st.write(answer)