from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os


load_dotenv()

documents = [
    "Nepal is a beautiful country in South Asia, famous for the Himalayas and Mount Everest. Its capital is Kathmandu, and its culture is rich in traditions and festivals.",

    "India is a large country in South Asia known for its diverse cultures, languages, and traditions. Its capital is New Delhi, and it is famous for the Taj Mahal.",

    "Japan is an island country in East Asia known for its advanced technology and beautiful landscapes. Its capital is Tokyo, and it is famous for cherry blossoms.",

    "China is one of the largest countries in the world, located in East Asia. Its capital is Beijing, and it is famous for the Great Wall.",

    "The United States is a large country in North America with 50 states. Its capital is Washington, D.C., and it is known for its technology and entertainment.",

    "The United Kingdom is a country in Europe made up of England, Scotland, Wales, and Northern Ireland. Its capital is London, famous for Big Ben.",

    "France is a European country famous for its art, fashion, and architecture. Its capital is Paris, which is home to the famous Eiffel Tower.",

    "Australia is a country and continent located in the Southern Hemisphere. Its capital is Canberra, and it is famous for kangaroos and koalas.",

    "Canada is a large country in North America known for its natural beauty and cold winters. Its capital is Ottawa, and it is famous for its lakes and forests.",

    "Brazil is the largest country in South America and is famous for its rainforests and football. Its capital is Brasília, and it is home to the Amazon rainforest."
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=32,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

question = "What is the capital of Brazil?"

question_embedding = embeddings.embed_query(question)
document_embeddings = embeddings.embed_documents(documents)

similarities = cosine_similarity([question_embedding], document_embeddings)[0]

index,sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[0]

print(f"Question: {question}")
print("Most similar document:")
print(f"Document: {documents[index]}")
print(f"Similarity Score: {sorted_similarities:.4f}")


