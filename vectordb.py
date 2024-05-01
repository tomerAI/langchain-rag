import os
import sys
from pathlib import Path
app_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(app_dir)) 
from api_keys import OPENAI_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY, ACTIVELOOP_TOKEN

# set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# set API key for LangSmith tracing, which will give us best-in-class observability.
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Langchain RAG"

# Deep Lake API
os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN


from langchain_community.document_loaders import TextLoader

root_dir = "libs"

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".py") and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception:
                pass
#print(f"{len(docs)}")


from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


from langchain_community.vectorstores import DeepLake

username = "thhjnissen"

db = DeepLake.from_documents(
    texts, embeddings, dataset_path=f"hub://{username}/langchain-code", overwrite=True
)
db


"""
from langchain_chroma import Chroma
db = Chroma.from_documents(documents, OpenAIEmbeddings())"""