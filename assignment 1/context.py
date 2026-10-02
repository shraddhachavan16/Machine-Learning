import subprocess


def ask_model(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3.2", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout


question = "What is a crossover point?"


# ==========================================
# EXPERIMENT 1 - LITTLE OR NO CONTEXT
# ==========================================

print("\n===== EXPERIMENT 1: LITTLE OR NO CONTEXT =====")

prompt1 = question

print("\nPrompt:")
print(prompt1)

answer1 = ask_model(prompt1)

print("\nOutput:")
print(answer1)


# ==========================================
# EXPERIMENT 2 - PROVIDED CONTEXT
# ==========================================

print("\n===== EXPERIMENT 2: PROVIDED CONTEXT =====")

# Read the context.txt provided with the assignment
with open("context.txt", "r", encoding="utf-8") as file:
    context = file.read()

prompt2 = context + "\n\nQuestion: " + question

print("\nContext:")
print(context)

print("\nQuestion:")
print(question)

answer2 = ask_model(prompt2)

print("\nOutput:")
print(answer2)


# ==========================================
# EXPERIMENT 3 - IMPROVED CONTEXT
# ==========================================

print("\n===== EXPERIMENT 3: IMPROVED / STRUCTURED CONTEXT =====")

context3 = """
Subject: Mathematics and Graphs

I am studying mathematical functions and graphs.

I have two curves on the same graph.
The curves can meet or cross each other.

I want to understand the point where the curves meet or cross.

Use this context to understand the question.
Give a simple and understandable explanation.
Give one real life example.

The intended meaning is related to mathematics and graphs,
not Genetic Algorithms.
"""

prompt3 = context3 + "\nQuestion: " + question

print("\nContext:")
print(context3)

print("\nQuestion:")
print(question)

answer3 = ask_model(prompt3)

print("\nOutput:")
print(answer3)