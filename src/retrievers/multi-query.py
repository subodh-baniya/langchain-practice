from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=128
)

llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
)

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding =embeddings,
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
    collection_name="my_docs_multi_query"
)

normal_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
    llm=llm
)

results = retriever.invoke("How to improve energy levels and maintain balance?")
print("multi-query retriever results:\n")

for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print("Page Content:", result.page_content)
    print("Source:", result.metadata.get("source"))
    print()

print("*"*150)  
print("Normal Retriever Results:\n")  

normal_results = normal_retriever.invoke("How to improve energy levels and maintain balance?")

for i, result in enumerate(normal_results):
    print(f"Normal Result {i+1}:")
    print("Page Content:", result.page_content)
    print("Source:", result.metadata.get("source"))
    print()