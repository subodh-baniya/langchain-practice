from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os


load_dotenv()

documents = [
    "The company was founded in 2018 and currently has 120 employees.",

    "The engineering department is responsible for developing and maintaining software products.",

    "Employees must submit their weekly progress reports every Friday.",

    "The marketing team manages social media campaigns and promotional activities.",

    "Project managers are responsible for assigning tasks and monitoring project progress.",

    "All employees must attend the monthly company meeting.",

    "The human resources department handles recruitment, employee onboarding, and leave requests.",

    "Employees receive 20 days of paid annual leave.",

    "The finance department manages budgets, invoices, and employee salaries.",

    "The IT department maintains company servers, networks, and security systems.",

    "Employees must use their company email for official communication.",

    "New employees must complete their onboarding process within the first week.",

    "The sales department is responsible for finding customers and managing client relationships.",

    "Employees working remotely must update their daily work status.",

    "The company uses Slack for internal team communication.",

    "Project deadlines must be approved by the assigned project manager.",

    "Employees can request leave through the HR management portal.",

    "The security team reviews access permissions every three months.",

    "The development team uses GitHub to manage source code and review changes.",

    "Customer complaints must be forwarded to the customer support department.",

    "The company conducts employee performance reviews every six months.",

    "Department managers must submit monthly budget reports to the finance team.",

    "All project documents must be stored in the company's shared Google Drive.",

    "Employees must enable two-factor authentication on company accounts.",

    "The product team collects customer feedback to improve existing features.",

    "Emergency maintenance must be reported to the IT department immediately.",

    "The company provides technical training to employees every quarter.",

    "Meeting minutes must be shared with all participants within 24 hours.",

    "Employees must inform their managers before working overtime.",

    "The customer support team operates from 9 AM to 5 PM, Monday through Friday."
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=32,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

models = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
)



question = input("Enter your question: ")

question_embedding = embeddings.embed_query(question)
document_embeddings = embeddings.embed_documents(documents)

similarities = cosine_similarity([question_embedding], document_embeddings)[0]

index,sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[0]

print(f"Most relevant document: {documents[index]}")

model_prompt = f"""
You are a helpful assistant. Based on the following document, answer the question below.
Document: {documents[index]}
Question: {question}
Answer:
"""

result = models.invoke(model_prompt)

print(result.content[0]["text"])




