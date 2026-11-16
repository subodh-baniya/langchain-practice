from langchain_core.tools import tool,StructuredTool
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: int = Field(description="The first number to add")
    b: int = Field(description="The second number to add")

@tool(args_schema=AddInput)
def add(a: int, b: int) ->  int:
    """Adds two numbers."""
    return a + b

result = add.invoke({"a": 5, "b": 10})
print(f"Result: {result}")

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

structured_multiply = StructuredTool.from_function(multiply, args_schema=MultiplyInput)

result = structured_multiply.invoke({"a": 5, "b": 10})
print(f"Result: {result}")

