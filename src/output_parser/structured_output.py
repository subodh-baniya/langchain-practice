from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser,
    ResponseSchema
)

from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

schema = [
    ResponseSchema(name="name", description="The name of the user"),
    ResponseSchema(name="age", description="The age of the user"),
    ResponseSchema(name="email", description="The email of the user"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    input_variables=[],
    template="""
    Generate one fictional user.

    Return ONLY a valid JSON object.
    Do not include explanations.
    Do not include markdown code blocks.

    Required fields:
    - name: string
    - age: string
    - email: string

    Example:
    {{
        "name": "John Smith",
        "age": "25",
        "email": "john@example.com"
    }}

    {format_instructions}
    """,
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = template | model | parser

result = chain.invoke({})

print(result)
print(type(result))

print(result["name"])
print(result["age"])
print(result["email"])