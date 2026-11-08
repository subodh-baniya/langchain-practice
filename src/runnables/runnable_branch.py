from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.2,
    max_output_tokens=1024,
    max_retries=3
)

class NumberOutput(BaseModel):
    number: int

parser = PydanticOutputParser(pydantic_object=NumberOutput)

prompt_template = PromptTemplate(
    template="You have to generate a number between 1 and 20 but \n {format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt_template_2 = PromptTemplate(
    template="You have to get tell me which number is generated with a positive signal : {number}",
    input_variables=["number"],
    
)

prompt_template_3 = PromptTemplate(
    template="You have to get tell me which number is generated with a negative signal : {number}",
    input_variables=["number"],
)

str_parser=StrOutputParser()

chain_2=prompt_template_2|model    
chain_3=prompt_template_3|model


branch_output = RunnableBranch(
    (RunnableLambda(lambda x: 0 < x.number < 10), chain_2),
    (RunnableLambda(lambda x: 10 <= x.number < 20), chain_3),
    RunnableLambda(lambda x: "This is okay")
)

chain=prompt_template|model|parser|branch_output|str_parser

result=chain.invoke({})

print(result)



