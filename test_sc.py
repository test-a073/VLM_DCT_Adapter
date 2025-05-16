import os
from openai import OpenAI

# Set your API key securely from an environment variable
api_key = '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'

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def ask_question(question, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for question answering."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example question
    user_question = input("Ask a question: ")
    answer = ask_question(user_question)
    print("Answer:", answer)
