from langchain_core.documents import Document 
from pathlib import Path
from pypdf import PdfReader

documents=[]

pdf_path=Path("src/files")

for pdf_file in pdf_path.glob("*.pdf"):

    reader=PdfReader(str(pdf_file))
    print(f"Processing {pdf_file} with {len(reader.pages)} pages.")

    for page_number, page in enumerate(reader.pages):

        text=page.extract_text() or ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_file),
                    "page": page_number
                }
            )
        )



  
print(documents)    


texts=[]

text_document=open("src/document_loader/hello.txt", "r", encoding="utf-8")

for line in text_document.readlines():
    texts.append(Document(
        page_content=line,
        metadata={
            "source": "src/document_loader/hello.txt"
        }
    ))

text_document.close()
print(texts)
