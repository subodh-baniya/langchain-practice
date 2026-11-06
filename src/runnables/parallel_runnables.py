from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

prompt_template1 = PromptTemplate(
    input_variables=["user_input"],
    template="""Create a oneline tweet for the following product: {user_input}""")

prompt_template2 = PromptTemplate(
    input_variables=["user_input"],
    template="""Create a one line linkedin post for the following product: {user_input}""")

parser = StrOutputParser()

chain=RunnableParallel({
    "tweet": prompt_template1 | model | parser,
    "linkedin_post": prompt_template2 | model | parser
})

result = chain.invoke({"user_input": "Sony WH-1000XM5"})

print(result["tweet"])
print(result["linkedin_post"])