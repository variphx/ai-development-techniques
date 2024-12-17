### RETRIEVE API KEYS FROM ENVIRONMENT ###

import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY is not None, "`GEMINI_API_KEY` is yet to be defined"

##########################################

### DEFINE GLOBAL SETTINGS ###

from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

GEMINI_LLM = Gemini(api_key=GEMINI_API_KEY)

GEMINI_EMBED_MODEL_NAME = "models/text-embedding-004"
GEMINI_EMBED_MODEL = GeminiEmbedding(
    model_name=GEMINI_EMBED_MODEL_NAME,
    api_key=GEMINI_API_KEY,
)

Settings.llm = GEMINI_LLM
Settings.embed_model = GEMINI_EMBED_MODEL

##############################

### LOAD DATA ###

from llama_index.core import SimpleDirectoryReader
from pathlib import Path

input_dir = Path.cwd().joinpath("data")
documents = SimpleDirectoryReader(input_dir=input_dir).load_data()

#################

### DEFINE INGESTION PIPELINE ###

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pathlib import Path

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("to-be-named")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

transformations = [
    TitleExtractor(),
    SemanticSplitterNodeParser(embed_model=GEMINI_EMBED_MODEL),
    GEMINI_EMBED_MODEL,
]

ingestion_pipeline = IngestionPipeline(
    transformations=transformations,
    vector_store=vector_store,
)

ingestion_pipeline.run(documents)

#################################

### RAG WITH INGESTION PIPELINE ###

from llama_index.core import VectorStoreIndex

ingestion_pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()

response = query_engine.query("Who is Little Red Riding Hood?")

print(response)

###################################
