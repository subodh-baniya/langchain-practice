from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    language="en",
)

query = "What is LangChain?"

results = retriever.invoke(query)

print(results[1].page_content)
