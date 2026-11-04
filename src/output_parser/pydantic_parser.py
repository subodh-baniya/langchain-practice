from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

class placeinfo(BaseModel):
    name: str = Field(description="The name of the place")
    location: str = Field( description="The location of the place")
    description: str = Field( description="A brief description of the place")
    age: int = Field( gt=18,description="The age of the place in years")

parser = PydanticOutputParser(pydantic_object=placeinfo)

template = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an expert travel guide.
Provide detailed information about the following fictional place:
{user_input}
Instructions:
- Include the name, location, and a brief description of the place.
- Provide the age of the place in years.
Output format:
{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"user_input": "A fictional place called 'Mystic Falls'."})

print(result)