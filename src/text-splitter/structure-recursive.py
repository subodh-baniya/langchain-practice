from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


text = """
def detect_language(text):
    if "def " in text or "import " in text:
        return "Python"
    elif "function " in text or "const " in text:
        return "JavaScript"
    elif "#include" in text:
        return "C/C++"
    else:
        return "Plain Text"
"""


def detect_language(text):
    try:
        lexer = guess_lexer(text)
        return lexer.name
    except ClassNotFound:
        return "text"


detected_language = detect_language(text)

print("Detected Language:", detected_language)

language_mapping = {
    "Python": Language.PYTHON,
    "JavaScript": Language.JS,
    "TypeScript": Language.TS,
    "Java": Language.JAVA,
    "C": Language.C,
    "C++": Language.CPP,
    "Go": Language.GO,
    "Rust": Language.RUST,
}

language = language_mapping.get(detected_language)

if language:
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=50,
        chunk_overlap=5
    )
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=5
    )

result = text_splitter.split_text(text)

print(result[0])