from langchain_community.document_loaders import PyPDFLoader


pdf_loader=PyPDFLoader("src/document_loader/pdf-1.pdf")

result=pdf_loader.load()

print(result)