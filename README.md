# Document Q&A API (RAG System)

A FastAPI-based intelligent document Q&A system that uses Retrieval-Augmented Generation (RAG) to answer questions based on uploaded documents. The system uses OpenAI for embeddings and chat completion, and Pinecone for vector storage.

## Features

- 📄 Upload and process text documents (.txt, .md)
- 🔍 Semantic search using OpenAI embeddings
- 💬 Intelligent question answering using GPT models
- 🗄️ Persistent vector storage with Pinecone
- 🚀 Fast and efficient document chunking
- ✅ Health monitoring and system status endpoints

---

## Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.10 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

You'll also need API keys for:
- **OpenAI API** - [Get your API key](https://platform.openai.com/api-keys)
- **Pinecone API** - [Get your API key](https://www.pinecone.io/)

---

## Step-by-Step Setup Guide

### 1. Clone or Navigate to the Project

```bash
cd /home/tnt/Desktop/code/ai/sfu_api_rag
```

Or if cloning from a repository:
```bash
git clone <your-repo-url>
cd sfu_api_rag
```

### 2. Create a Virtual Environment

Creating a virtual environment isolates your project dependencies:

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

**On Linux/MacOS:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` prefix in your terminal prompt.

### 4. Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

Or if using `uv` (faster alternative):
```bash
uv pip install -r requirements.txt
```

Alternatively, you can install directly from `pyproject.toml`:
```bash
pip install -e .
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
touch .env
```

Open the `.env` file and add your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
```

**Important Notes:**
- Replace `your_pinecone_api_key_here` with your actual Pinecone API key
- Replace `your_pinecone_index_name_here` with your Pinecone index name (e.g., "quickstart-js")
- Replace `your_openai_api_key_here` with your actual OpenAI API key
- The `OPENAI_CHAT_MODEL` is optional (defaults to gpt-5.1 in code, but you can use gpt-4o-mini for cost efficiency)

### 6. Create Required Directories

The application will create the `data` directory automatically, but you can create it manually:

```bash
mkdir -p data
```

### 7. Set Up Pinecone Index

Before running the application, make sure you have created a Pinecone index:

1. Log in to [Pinecone Console](https://app.pinecone.io/)
2. Create a new index with these settings:
   - **Dimension**: 1024
   - **Metric**: Cosine
   - **Cloud**: Your preferred cloud provider
3. Copy the index name to your `.env` file

---

## Running the Application

### Start the Server

Run the FastAPI application using uvicorn:

```bash
python main.py
```

Or alternatively:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The `--reload` flag enables auto-reload during development (optional).

### Verify the Server is Running

You should see output like:

```
✅ Student Enquiry Assistant system initialized successfully!
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

The API will be available at:
- **API Base URL**: http://localhost:8001
- **API Documentation (Swagger)**: http://localhost:8001/docs
- **Alternative Documentation (ReDoc)**: http://localhost:8001/redoc

---

## Using the API

### 1. Check System Status

```bash
curl http://localhost:8001/
```

**Response:**
```json
{
  "status": "ready",
  "vector_store_connected": true,
  "current_documents": [],
  "total_chunks": 0,
  "pinecone_stats": {
    "total_vectors": 0,
    "index_fullness": 0,
    "dimension": 1024
  }
}
```

### 2. Upload a Document

```bash
curl -X POST "http://localhost:8001/upload" \
  -F "file=@/path/to/your/document.txt"
```

**Response:**
```json
{
  "message": "Document uploaded and processed successfully",
  "filename": "document.txt",
  "chunks_created": 15,
  "timestamp": "2026-02-23T10:30:00.123456"
}
```

### 3. Ask a Question

```bash
curl -X POST "http://localhost:8001/question/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "status": "success",
  "message": "Answer generated successfully",
  "question": "What is the main topic of the document?",
  "answer": "Based on the document, the main topic is...",
  "relevant_chunks": 3,
  "timestamp": "2026-02-23T10:35:00.123456"
}
```

### 4. List All Documents

```bash
curl http://localhost:8001/documents
```

### 5. Health Check

```bash
curl http://localhost:8001/health
```

### 6. Clear All Documents

```bash
curl -X DELETE http://localhost:8001/clear
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Get system status and statistics |
| POST | `/upload` | Upload a text document (.txt, .md) |
| POST | `/question/stream` | Ask a question about uploaded documents |
| GET | `/documents` | List all uploaded documents |
| GET | `/health` | Health check endpoint |
| DELETE | `/clear` | Clear all documents from vector store |

---

## Interactive API Documentation

Once the server is running, visit:

**Swagger UI**: [http://localhost:8001/docs](http://localhost:8001/docs)

This provides an interactive interface where you can:
- Test all API endpoints
- View request/response schemas
- Execute API calls directly from the browser

---

## Project Structure

```
sfu_api_rag/
├── main.py                 # Main FastAPI application
├── pyproject.toml         # Project dependencies and metadata
├── .env                   # Environment variables (create this)
├── data/                  # Uploaded documents and state files
│   ├── system_state.json  # Persisted system state
│   └── *.txt             # Uploaded documents
├── .venv/                # Virtual environment (created by you)
└── README.md             # This file
```

---

## Troubleshooting

### Issue: Module not found errors

**Solution:** Make sure your virtual environment is activated and dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Missing required environment variables"

**Solution:** Check your `.env` file contains all required keys:
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `OPENAI_API_KEY`

### Issue: "Could not sync with Pinecone"

**Solution:**
1. Verify your Pinecone API key is correct
2. Ensure your Pinecone index exists and has dimension 1024
3. Check your internet connection

### Issue: Port 8001 already in use

**Solution:** Either stop the process using that port or run on a different port:
```bash
python main.py --port 8002
```

Or:
```bash
uvicorn main:app --port 8002
```

---

## Development Tips

### Running in Development Mode with Auto-Reload

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Viewing Logs

The application logs important events to the console. Watch for:
- ✅ Success indicators
- ⚠️ Warnings
- ❌ Errors

### Testing the API

Use the Swagger UI at http://localhost:8001/docs for easy testing, or use tools like:
- **curl** (command line)
- **Postman** (GUI)
- **HTTPie** (command line)

---

## Cost Considerations

This application uses paid APIs:

- **OpenAI API**: Charges for embeddings and chat completions
- **Pinecone**: Free tier available, paid plans for production use

**Tips to minimize costs:**
- Use `gpt-4o-mini` instead of `gpt-5.1` for chat (set in `.env`)
- Process smaller documents during development
- Use Pinecone's free tier for testing

---

## Security Notes

⚠️ **Never commit your `.env` file to version control!**

Add `.env` to your `.gitignore`:
```bash
echo ".env" >> .gitignore
```

For production deployment:
- Use environment variables instead of `.env` files
- Restrict CORS origins (currently set to allow all)
- Add authentication/authorization
- Use HTTPS

---

## Dependencies

Main dependencies (see `pyproject.toml` for full list):
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` - LLM orchestration
- `langchain-openai` - OpenAI integration
- `langchain-pinecone` - Pinecone vector store
- `python-dotenv` - Environment variable management
- `pinecone` - Vector database client

---

## License

[Add your license here]

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Open an issue in the repository

---

## Next Steps

After successfully running the application:

1. ✅ Upload a sample document via `/upload`
2. ✅ Ask questions via `/question/stream`
3. ✅ Integrate with your frontend application
4. ✅ Customize the prompt template in `main.py` (lines 148-159)
5. ✅ Add authentication for production use
6. ✅ Configure proper CORS settings

Happy coding! 🚀
