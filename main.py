import ollama
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

# see: https://github.com/ollama/ollama-python/tree/main
# ToDo: 
# git
# set up ruff linting

DIR = "~/Documents/nauka/ML_Literatur/paper/"

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

# llamaindex with ollama docs:
# https://docs.llamaindex.ai/en/stable/examples/llm/ollama/

reader = SimpleDirectoryReader(input_dir=DIR)  # maybe use llama-parse instead of simpledirectory reader
docs = reader.load_data()
print(docs[0])

# llamaindex:
# advanced rag example:
# https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb