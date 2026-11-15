from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from chunk import get_chunks
import os
load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=128
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    collection_name="my_documents"
)

chunks = get_chunks()

vectorstore.add_documents(chunks)

retriver = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_retriever():
    return retriver
