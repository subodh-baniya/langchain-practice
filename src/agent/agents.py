from langchain.agents import  create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
load_dotenv()

llm = ChatGoogleGenerativeAI(
      model="gemini-3.5-flash-lite"
)

search = DuckDuckGoSearchRun()

agent = create_agent(model=llm, tools=[search], system_prompt="You are a helpful assistant that can answer questions ")

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is langchain?"
        }
    ]
})

print(response["messages"][-1].content[0]["text"])