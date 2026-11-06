from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

prompt_template1 = PromptTemplate(
    input_variables=["user_input"],
    template="""Create a joke on the topic: {user_input}""")

parser = StrOutputParser()

def wordCount(text):
    return len(text.split())

chain=RunnableSequence(prompt_template1 , model , parser)

parallel_chain=RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(wordCount)
})

final_chain=RunnableSequence(chain,parallel_chain)

result = final_chain.invoke({"user_input": "chicken"})

print(result)