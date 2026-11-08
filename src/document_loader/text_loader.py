from langchain_community.document_loaders import TextLoader

text_loader=TextLoader("src/document_loader/hello.txt",encoding="utf-8")

result=text_loader.load()

print(result[0].metadata)