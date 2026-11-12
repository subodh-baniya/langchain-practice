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
        page_content=(
            "To fix a slow Python script, use cProfile to find bottlenecks and"
            " optimize loops."
        ),
        metadata={"id": 1, "type": "exact_match"},
    ),
    Document(
        page_content=(
            "Fixing a slow Python script starts with profiling your code using"
            " Profile to locate slow loops."
        ),
        metadata={"id": 2, "type": "near_duplicate"},
    ),
    Document(
        page_content=(
            "If your Python script is running slow, run the cProfile module to"
            " identify performance bottlenecks."
        ),
        metadata={"id": 3, "type": "near_duplicate"},
    ),
    Document(
        page_content=(
            "Replace standard Python lists with NumPy arrays to speed up heavy"
            " mathematical computations."
        ),
        metadata={"id": 4, "type": "relevant_diverse"},
    ),
    Document(
        page_content=(
            "Use multiprocessing or asyncio to run I/O-bound and CPU-bound"
            " tasks concurrently in Python."
        ),
        metadata={"id": 5, "type": "relevant_diverse"},
    ),
    Document(
        page_content=(
            "Caching expensive function calls with functools.lru_cache can"
            " drastically reduce Python execution time."
        ),
        metadata={"id": 6, "type": "relevant_diverse"},
    ),
    Document(
        page_content=(
            "PyPy is an alternative Just-In-Time (JIT) compiler that runs pure"
            " Python code much faster than CPython."
        ),
        metadata={"id": 7, "type": "slightly_relevant"},
    ),
    Document(
        page_content=(
            "Python 3.13 introduced experimental support for disabling the"
            " Global Interpreter Lock (GIL)."
        ),
        metadata={"id": 8, "type": "loosely_related"},
    ),
    Document(
        page_content=(
            "How to install Python and set up a virtual environment on Ubuntu"
            " Linux."
        ),
        metadata={"id": 9, "type": "noise"},
    ),
    Document(
        page_content=(
            "JavaScript promises and async/await help prevent blocking the main"
            " browser thread."
        ),
        metadata={"id": 10, "type": "noise"},
    ),
]



vectorstore = Chroma.from_documents(
    embedding=embeddings,
    documents=documents,
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    collection_name="my_docs_mmr"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,"lambda_mult": 0.5}
    )

query = "How to optimize a slow Python script?"

results = retriever.invoke(query)

for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print("Page Content:", result.page_content)
    print()

print(results)

    