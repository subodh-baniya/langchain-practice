from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=32
)

chunker = SemanticChunker(
    embeddings=embeddings,
    min_chunk_size=10,
    breakpoint_threshold_type="percentile",
)


text = """
Python is a programming language. It is widely used for web development.
Python also supports machine learning and data science.

The Earth revolves around the Sun. The Moon revolves around the Earth.
The solar system contains eight planets."""

result = chunker.split_text(text)
print(result)
