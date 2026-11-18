from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)

@tool
def multiply_numbers(num1: float, num2: float) -> float:
    """Multiplies two numbers and returns the result."""
    return num1 * num2


llm_with_tool = llm.bind_tools([multiply_numbers])

docs = []

query = "Please multiply 7 and 8."

docs.append(HumanMessage(content=query))

llm_response = llm_with_tool.invoke(docs)

docs.append(llm_response)

for tool_call in llm_response.tool_calls:

    if tool_call["name"] == "multiply_numbers":

        result = multiply_numbers.invoke(tool_call["args"])

        docs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

final_response = llm_with_tool.invoke(docs)

docs.append(final_response)

print(final_response.content)