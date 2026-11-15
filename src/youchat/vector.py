from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from chunk import get_chunks
from ytload import get_video_id
import os
load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=128
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./db",
    collection_name="my_documents"
)

chunks = get_chunks()

vectorstore.add_documents(chunks)

retriver = vectorstore.as_retriever(
    search_kwargs={"k": 3},
    filter={"video_id": get_video_id()}
    )

def get_retriever():
    return retriver
