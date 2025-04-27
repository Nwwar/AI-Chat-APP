from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import math
import os

app = FastAPI()

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up the OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize ChromaDB client with OpenAI embeddings
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    "documents",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
)

# Initialize LLM with OpenAI API key
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


# Calculator Tool
def calculator(expression: str) -> str:
    try:
        if "square root" in expression.lower():
            num = float(expression.lower().replace("square root of", "").strip())
            return str(math.sqrt(num))
        return str(eval(expression))  # Simple eval for basic arithmetic
    except Exception as e:
        return f"Error calculating: {str(e)}"


tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="A tool to perform basic arithmetic operations including square root."
    )
]

# Initialize LangChain agent
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)


# Helper function to chunk text
def chunk_text(text: str, chunk_size: int = 1000) -> list:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt or .pdf")

    content = ""
    if file.filename.endswith('.txt'):
        content = await file.read()
        content = content.decode('utf-8')
    elif file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            content += page.extract_text()

    # Chunk the document
    chunks = chunk_text(content)

    # Store chunks in ChromaDB
    collection.add(
        documents=chunks,
        ids=[f"{file.filename}_{i}" for i in range(len(chunks))]
    )

    return {"message": f"File {file.filename} uploaded and processed successfully"}


@app.post("/chat")
async def chat(query: dict):
    user_query = query.get("question", "")
    if not user_query:
        raise HTTPException(status_code=400, detail="No question provided")

    # Perform vector search in ChromaDB
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )

    # Extract relevant chunks
    context = " ".join(results["documents"][0]) if results["documents"] else ""

    # Prepare prompt for RAG
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Based on the following context: {context}\nAnswer the question: {question}"
    )
    prompt = prompt_template.format(context=context, question=user_query)

    # Run the agent with the prompt
    response = agent.run(prompt)

    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
