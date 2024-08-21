import ollama
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import os

# see: https://github.com/ollama/ollama-python/tree/main
# ToDo: 
# git
# set up ruff linting

DIR = "~/Documents/nauka/ML_Literatur/paper/"

load_dotenv()
MODEL = os.getenv("MODEL")

llm = Ollama(model=MODEL, request_timeout=120.0)
resp = llm.complete("Who is Paul Graham?")
print(resp)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# response = ollama.chat(model=MODEL, messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?'
#   },
# ],
# options={
#     "temperature": 1.5
# }
# )

# print(response['message']['content'])
# print(ollama.list())

# llamaindex with ollama docs:
# https://docs.llamaindex.ai/en/stable/examples/llm/ollama/

reader = SimpleDirectoryReader(input_dir=DIR)  # maybe use llama-parse instead of simpledirectory reader
docs = reader.load_data()
print(docs[0])

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("Who wrote the playing Atari with Deep Learning Paper?")
print(response)

# llamaindex:
# advanced rag example:
# https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb