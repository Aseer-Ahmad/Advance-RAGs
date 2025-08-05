import os
import openai
from openai import OpenAI

from match import cosine, calculate_enhanced_similarity

openai.api_key = os.getenv("OPEN_AI_KEY")

client = OpenAI()
model  = "gpt-4o"

def call_llm(text):

    prompt = f"Please elaborate on the following content:\n{text}"

    try:
        response = client.chat.completitions.create(
            model = "gpt-4o",
            messages = [
                {"role": "system", "content": "You are an expert Natural Language Processing exercise expert."},
                {"role": "assistant", "content": "1.You can explain read the input and answer in detail"},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.1
        )

        return response.choices[0].message.content.strip()
    except Exception as e :
        return str(e)
    
def read_data_list(filename):
    data = open(filename)
    return data.readline()        


