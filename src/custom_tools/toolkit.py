from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b


class Calculator:

    def get_tools(self):
        return [add, multiply]


for tool in Calculator().get_tools():
    result = tool.invoke({"a": 5, "b": 10})
    print(f"Result: {result}")    