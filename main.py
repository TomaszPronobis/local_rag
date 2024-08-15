import ollama
from dotenv import load_dotenv
import os

# see: https://github.com/ollama/ollama-python/tree/main
# ToDo: 
# git
# set up ruff linting

load_dotenv()
MODEL = os.getenv("MODEL")

response = ollama.chat(model=MODEL, messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
],
options={
    "temperature": 1.5
}
)

# print(response['message']['content'])
print(ollama.list())