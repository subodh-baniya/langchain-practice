from langchain_docling import DoclingLoader
from pathlib import Path
from langchain_core.documents import Document
from langchain_docling.loader import ExportType

documents = []


pdf_path = Path("src/files")

for pdf_file in pdf_path.glob("*.pdf"):

    loader = DoclingLoader(
        file_path=str(pdf_file),
        export_type=ExportType.MARKDOWN,
    )

    document = loader.lazy_load()

    for doc in document:
        documents.append(
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
        )



print(documents[0].page_content)        