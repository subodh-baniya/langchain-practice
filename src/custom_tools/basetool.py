from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class multiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

class MultiplyTool(BaseTool):
    name : str = "multiply"
    description :str = "Multiplies two numbers."
    args_schema : type[multiplyInput] = multiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b 



multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 5, "b": 10})
print(f"Result: {result}")

