from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=128
)

documents = [
    Document(   
        page_content="Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, automation, artificial intelligence, machine learning, and data science. Python supports multiple programming paradigms, including object-oriented and functional programming.",
        metadata={"source": "python_guide.pdf", "page": 1}
    ),
    Document(
        page_content="Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning.",
        metadata={"source": "machine_learning.pdf", "page": 1}
    ),
    Document(
        page_content="Vector databases are designed to store and search high-dimensional vector embeddings. They are particularly useful in semantic search applications because they can retrieve information based on meaning rather than exact keyword matches. Popular vector databases include Pinecone, Chroma, Weaviate, and Milvus.",
        metadata={"source": "vector_databases.pdf", "page": 5}
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by large language models. It provides integrations for language models, embedding models, document loaders, text splitters, vector stores, and retrieval systems. Developers can combine these components to build retrieval-augmented generation applications.",
        metadata={"source": "langchain.pdf", "page": 3}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. First, documents are divided into smaller chunks and converted into embeddings. These embeddings are stored in a vector database. When a user asks a question, the system retrieves relevant chunks and provides them as context to a language model. The model then generates an answer grounded in the retrieved information.",
        metadata={"source": "rag_guide.pdf", "page": 10}
    ),
]

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    collection_name="my_documents"
)

vectorstore.add_documents(documents)

search_results= vectorstore.get(include=["metadatas", "embeddings", "documents"])

semantic_search_results = vectorstore.similarity_search("What is LangChain?", k=2)

print(semantic_search_results)


