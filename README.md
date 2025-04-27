# AI Chat Project

## Description

This project implements a web-based AI chatbot with Retrieval-Augmented Generation (RAG) capabilities. Users can interact with the chatbot through a simple web interface, ask questions, and upload documents (.txt, .pdf) to provide context for the AI's responses. The backend leverages FastAPI, Langchain, OpenAI, and ChromaDB to process documents, manage embeddings, and generate intelligent replies. The chatbot also includes a basic calculator tool.

## Features

* **Web-based Chat Interface:** Simple and clean UI for interacting with the chatbot (`chatbot.html`).
* **File Upload:** Supports uploading `.txt` and `.pdf` files to provide context.
* **Retrieval-Augmented Generation (RAG):** Uses uploaded document content stored in a vector database (ChromaDB) to provide contextually relevant answers.
* **OpenAI Integration:** Utilizes OpenAI's models for embeddings and chat completions (`gpt-4o`).
* **Langchain Agent:** Employs a Langchain agent to orchestrate the interaction between the language model, context retrieval, and tools.
* **Calculator Tool:** Includes a basic tool for performing arithmetic calculations (including square root).
* **FastAPI Backend:** Robust and asynchronous backend framework (`main.py`).

## Technologies Used

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python, FastAPI
* **AI/LLM:** OpenAI (GPT-4o), Langchain
* **Vector Database:** ChromaDB (in-memory)
* **PDF Processing:** PyPDF2
* **Dependencies:** See `requirements.txt`

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Set up Python Environment:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the project root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY='your_openai_api_key_here'
    ```
    *Note: You need a valid API key from OpenAI.*

## Running the Application

1.  **Start the Backend Server:**
    Run the FastAPI application using Uvicorn:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The server will be running at `http://localhost:8000`.

2.  **Open the Frontend:**
    Open the `chatbot.html` file directly in your web browser (e.g., double-click the file or use `File > Open` in your browser).

## Usage

1.  **Chat:** Type your questions or messages into the input field at the bottom and press Enter or click "Send".
2.  **Upload Files:** Click the paperclip icon (📎) to select a `.txt` or `.pdf` file. The file content will be processed and stored for context retrieval. A system message will confirm successful upload.
3.  **Contextual Answers:** Ask questions related to the content of the uploaded documents. The chatbot will use the relevant information to formulate its response.
4.  **Calculator:** Ask the chatbot to perform basic calculations (e.g., "What is 5 * 8?", "What is the square root of 16?").

## API Endpoints

* `POST /upload`: Handles file uploads. Expects multipart/form-data with a 'file' field.
* `POST /chat`: Handles chat messages. Expects JSON payload with a 'question' field (e.g., `{"question": "Your query here"}`).

## Notes

* The ChromaDB vector store is currently configured to run in-memory. This means uploaded document data will be lost when the backend server restarts. For persistence, ChromaDB needs to be configured differently.
* The calculator tool uses Python's `eval()` for basic arithmetic, which should be used cautiously in production environments due to potential security risks if input is not properly sanitized (though the current implementation is relatively contained within the agent).

