# import streamlit as st
# from google import genai
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=api_key)
# st.title("AI Storyteller Chatbot")
# st.write("Enter an idea and Gemini will create a fantasy story!")

# idea = st.text_input("Enter your story idea:",placeholder="Example: A boy finds a magical sword")
# if st.button(" Create Story"):

#     if idea:
#         prompt = f"""
#         You are a creative fantasy storyteller.
#         Create a fictional fantasy story based on this idea:
#         {idea}
#         Requirements:
#         - Give the story a title.
#         - Use simple English.
#         - Make the story interesting.
#         - Include a beginning, middle and ending.
#         - Keep the story around 300 words.
#         """
#         response = client.models.generate_content(model="gemini-3.6-flash",contents=prompt)
#         st.subheader("Your Story")
#         st.write(response.text)
# else:
#         st.warning("Please enter a story idea.")





from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

print("================================")
print("      AI STORYTELLER CHATBOT")
print("================================")

idea = input("Enter your story idea: ")

prompt = f"""
Create a simple fictional fantasy story about:

{idea}

Give the story a title.
Use simple English.
Make it interesting.
"""

chat = client.chats.create(
    model="gemini-3.5-flash-lite"
)

response = chat.send_message(prompt)

print("\n========== YOUR STORY ==========\n")
print(response.text)