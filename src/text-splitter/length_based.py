from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from pathlib import Path
from langchain_docling.loader import ExportType

pdf_path = Path("src/files")
chunk_documents = []

for pdf_file in pdf_path.glob("*.pdf"):
    loader = DoclingLoader(
        file_path=str(pdf_file),
        export_type=ExportType.MARKDOWN,
    )

    document = loader.lazy_load()

    for doc in document:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_index": i}
            )
            chunk_documents.append(new_doc)


print(chunk_documents)            