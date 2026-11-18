from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

@tool
def muliply_numbers(num1: float, num2: float) -> float:
    """Multiplies two numbers and returns the result as a string."""
    result = num1 * num2
    return f"The result of multiplying {num1} and {num2} is {result}."

llm_with_tool = llm.bind_tools([muliply_numbers])

result = llm_with_tool.invoke("Please multiply 5 and 10.")

print(result)
