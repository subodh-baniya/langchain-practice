from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal,Optional

load_dotenv()


model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1
)


json_schema={
    "type": "object",
    "title": "ModelOutput",
    "properties": {
        "summary": {
            "type": "string",
            "description": "consise summary of the product review"
        },
        "sentiment": {
            "type": "string",
            "description": "sentiment of the product review, either positive, negative or neutral"
        },
        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "pros of the product review"
        },
        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "cons of the product review"
        },
        "total_tokens": {
            "type": "integer",
            "description": ""
        }
    },
    "required": ["summary", 'sentiment', 'pros', 'cons', 'total_tokens']
}


structured_model=model.with_structured_output(json_schema)

result=structured_model.invoke("The Sony WH-1000XM5 delivers an impressive combination of premium sound quality, adaptive noise cancellation, and ergonomic design. Its ability to isolate ambient noise makes it particularly effective in crowded environments, while the balanced audio profile provides a rich listening experience across different genres. However, the non-foldable design compromises portability, and the premium price may not justify the upgrade for users who already own high-end headphones. Overall, it is a well-engineered product that excels in performance but has a few practical limitations.")

print(result)
