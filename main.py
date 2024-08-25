from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os

# see: https://github.com/ollama/ollama-python/tree/main
# ToDo: 
# git
# set up ruff linting

load_dotenv()
MODEL = os.getenv("MODEL")
DIR = os.getenv("DIR")

llm = Ollama(model=MODEL, request_timeout=120.0)
# resp = llm.complete("Who is Paul Graham?")
# print(resp)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# consider using optimum & insatruct embeddings: 
# https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/

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

reader = SimpleDirectoryReader(input_dir=DIR, recursive=False)  # maybe use llama-parse instead of simpledirectory reader
docs = reader.load_data()
# print(docs[0])

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
query = "Explain if Tomasz Pronobis would be a good fit for the open position at DeepSpin!"
print(query)
response = query_engine.query(query)
print(response)

# llamaindex:
# advanced rag example:
# https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb