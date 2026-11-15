from vector import get_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

load_dotenv()

retriever = get_retriever()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

template = """
You are a helpful assistant that answers questions based on the context provided.
Context:
{context}
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

parser = StrOutputParser()

while True:
    query = input("You:")
    if query.lower() == 'exit':
        break

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = prompt_template.format(context=context, question=query)
    response = model.invoke(prompt)
    answer = parser.parse(response)

    print(f"Bot: {answer.content[0]["text"]}")







