from langchain_text_splitters import RecursiveCharacterTextSplitter
from ytload import documents
from langchain_core.documents import Document

documents = documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

split = text_splitter.split_text(documents)

chunks = []

for i, chunk in enumerate(split):
    new_doc = Document(
        page_content=chunk,
        metadata={"chunk_index": i}
    )
    chunks.append(new_doc)

def get_chunks():
    return chunks

