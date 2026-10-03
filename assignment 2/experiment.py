# import streamlit as st
# import ollama

# # Page title
# st.title(" My Local Chatbot")

# st.write("Powered by Ollama and Llama 3.2")

# # Create chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []


# # Clear chat button
# if st.button("🧹 Clear Chat"):
#     st.session_state.messages = []
#     st.rerun()


# # Display old messages
# for message in st.session_state.messages:

#     if message["role"] == "user":

#         st.markdown("###  You")
#         st.write(message["content"])

#     else:

#         st.markdown("###  AI")
#         st.write(message["content"])


# # Get message from user
# user_message = st.chat_input("Type your message...")


# # If user entered a message
# if user_message:

#     # Display user message
#     st.markdown("### You")
#     st.write(user_message)

#     # Save user message
#     st.session_state.messages.append({
#         "role": "user",
#         "content": user_message
#     })

#     # Send message to Ollama
#     response = ollama.chat(
#         model="llama3.2",
#         messages=st.session_state.messages
#     )

#     # Get AI answer
#     ai_message = response["message"]["content"]

#     # Display AI answer
#     st.markdown("###  AI")
#     st.write(ai_message)

#     # Save AI answer
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": ai_message
import ollama

model = "llama3.2"

question = input("Enter your question: ")

with open("context.txt", "r") as file:
    context = file.read()


print("\n--- Experiment 1: No Context ---")

response = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": question}
    ]
)

print(response["message"]["content"])


print("\n--- Experiment 2: Relevant Context ---")

prompt = context + "\n\nQuestion: " + question

response = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response["message"]["content"])


print("\n--- Experiment 3: Structured Context ---")

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Give a simple and clear answer based on the context.
"""

response = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response["message"]["content"])