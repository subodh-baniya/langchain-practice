from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

directory = Path("src/files")

documents = []

for file_path in directory.rglob("*.pdf"):

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "page": page_number
                }
            )
        )

print(documents)