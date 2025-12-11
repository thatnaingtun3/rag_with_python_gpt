from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import shutil
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Document Q&A API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=3)

# State file path
STATE_FILE = "data/system_state.json"


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QuestionResponse(BaseModel):
    # status can be "success" or "error"
    status: str
    message: str
    question: str
    answer: str
    relevant_chunks: int
    timestamp: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    timestamp: str


class StatusResponse(BaseModel):
    status: str
    vector_store_connected: bool
    current_documents: List[str]
    total_chunks: int
    pinecone_stats: Optional[Dict]


# Global system instance
class QASystem:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.chat_model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.chat_model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1")
# gpt-5.1

        if not all([self.pinecone_api_key, self.pinecone_index, self.openai_api_key]):
            raise ValueError("Missing required environment variables")

        self.vector_store = None
        self.model = None
        self.retriever = None
        self.chain = None
        self.current_documents = []
        self.total_chunks = 0
        self.is_processing = False
        self.pinecone_client = None

        # Initialize on startup
        self._initialize_system()

        # Load persisted state
        self._load_state()

        # Check Pinecone for existing data
        self._sync_with_pinecone()

    def _initialize_system(self):
        """Initialize all components"""
        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)

            # Setup embeddings
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                dimensions=1024,
            )

            # Connect to vector store
            self.vector_store = PineconeVectorStore(
                index_name=self.pinecone_index, embedding=embeddings
            )

            # Setup chat model
            self.model = ChatOpenAI(
                model=self.chat_model_name,
                temperature=0.1,
                openai_api_key=self.openai_api_key,
            )

            # Setup retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )

            # Setup Q&A chain
            template = """
You are an expert assistant helping students by providing detailed responses based on the uploaded documents.

Below are relevant sections extracted from the documents:
{context}

Student's Enquiry: {question}

Please provide a clear, formal, and comprehensive response based on the information provided in the document sections above. Respond in plain sentences only; do not use markdown, bullet points, or special formatting. If the answer cannot be found within the provided text, kindly state that explicitly in plain text.

Response:
"""

            prompt = ChatPromptTemplate.from_template(template)
            self.chain = prompt | self.model

            print("✅ Student Enquiry Assistant system initialized successfully!")

        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            raise

    def _save_state(self):
        """Save current state to file"""
        try:
            os.makedirs("data", exist_ok=True)
            state = {
                "current_documents": self.current_documents,
                "total_chunks": self.total_chunks,
                "last_updated": datetime.now().isoformat(),
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
            print(
                f"💾 State saved: {len(self.current_documents)} documents, {self.total_chunks} chunks"
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not save state: {str(e)}")

    def _load_state(self):
        """Load persisted state from file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.current_documents = state.get("current_documents", [])
                self.total_chunks = state.get("total_chunks", 0)
                print(
                    f"📂 State loaded: {len(self.current_documents)} documents, {self.total_chunks} chunks"
                )
                print(f"   Last updated: {state.get('last_updated', 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load state: {str(e)}")

    def _sync_with_pinecone(self):
        """Check Pinecone index for existing data"""
        try:
            # Get index stats
            index = self.pinecone_client.Index(self.pinecone_index)
            stats = index.describe_index_stats()

            # Update total chunks based on Pinecone data
            vector_count = stats.get("total_vector_count", 0)

            if vector_count > 0:
                print(f"🔄 Found {vector_count} vectors in Pinecone index")

                # If we have vectors but no local state, update state
                if not self.current_documents and vector_count > 0:
                    self.current_documents = [
                        "[Previous documents detected in Pinecone]"
                    ]
                    self.total_chunks = vector_count
                    print(
                        f"✅ Synchronized with Pinecone: {vector_count} existing vectors found"
                    )
                    self._save_state()
                elif vector_count != self.total_chunks:
                    print(
                        f"⚠️ Warning: Local state ({self.total_chunks} chunks) differs from Pinecone ({vector_count} vectors)"
                    )
                    self.total_chunks = vector_count
                    self._save_state()
            else:
                print("📭 Pinecone index is empty")
                if self.current_documents:
                    print(
                        "⚠️ Warning: Local state shows documents but Pinecone is empty. Clearing local state."
                    )
                    self.current_documents = []
                    self.total_chunks = 0
                    self._save_state()

        except Exception as e:
            print(f"⚠️ Warning: Could not sync with Pinecone: {str(e)}")

    def get_pinecone_stats(self):
        """Get current Pinecone index statistics"""
        try:
            index = self.pinecone_client.Index(self.pinecone_index)
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "dimension": stats.get("dimension", 1024),
            }
        except Exception as e:
            print(f"Error getting Pinecone stats: {str(e)}")
            return None

    def create_text_chunks(self, content: str, filename: str) -> List[Document]:
        """Split text into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_text(content)
        documents = []

        for i, chunk in enumerate(chunks):
            first_line = (
                chunk.split("\n")[0][:60] + "..."
                if len(chunk.split("\n")[0]) > 60
                else chunk.split("\n")[0]
            )

            document = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "title": first_line,
                    "word_count": len(chunk.split()),
                    "upload_timestamp": datetime.now().isoformat(),
                },
                id=f"{filename}_chunk_{i}",
            )
            documents.append(document)

        return documents

    async def process_document(self, content: str, filename: str):
        """Process and store document embeddings"""
        self.is_processing = True
        try:
            # Create chunks
            documents = self.create_text_chunks(content, filename)
            chunks_created = len(documents)

            # Create embeddings and add to vector store
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key,
                dimensions=1024,
            )

            # Add documents to existing vector store
            await asyncio.get_event_loop().run_in_executor(
                executor, self.vector_store.add_documents, documents
            )

            # Update state
            if filename not in self.current_documents:
                self.current_documents.append(filename)
            self.total_chunks += chunks_created

            # Save state to file
            self._save_state()

            return chunks_created

        finally:
            self.is_processing = False

    async def ask_question(self, question: str, top_k: int = 3) -> dict:
        """Ask a question and get an answer"""
        if not all([self.retriever, self.chain]):
            raise ValueError("System not properly initialized")

        # Update retriever with new k value
        self.retriever.search_kwargs = {"k": top_k}

        # Get relevant chunks and generate answer
        context_docs = await asyncio.get_event_loop().run_in_executor(
            executor, self.retriever.invoke, question
        )

        result = await asyncio.get_event_loop().run_in_executor(
            executor, self.chain.invoke, {"context": context_docs, "question": question}
        )

        return {"answer": result.content, "relevant_chunks": len(context_docs)}

    async def clear_all_data(self):
        """Clear all data from Pinecone and reset state"""
        try:
            # Clear Pinecone index
            index = self.pinecone_client.Index(self.pinecone_index)
            index.delete(delete_all=True)

            # Reset state
            self.current_documents = []
            self.total_chunks = 0

            # Save cleared state
            self._save_state()

            return True
        except Exception as e:
            print(f"Error clearing data: {str(e)}")
            return False


# Initialize system
qa_system = QASystem()


# API Routes
@app.get("/", response_model=StatusResponse)
async def root():
    """Get system status"""
    pinecone_stats = qa_system.get_pinecone_stats()
    return StatusResponse(
        status="ready" if not qa_system.is_processing else "processing",
        vector_store_connected=qa_system.vector_store is not None,
        current_documents=qa_system.current_documents,
        total_chunks=qa_system.total_chunks,
        pinecone_stats=pinecone_stats,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """Upload and process a text document"""

    # Validate file type
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400, detail="Only .txt and .md files are supported"
        )

    # Check if system is already processing
    if qa_system.is_processing:
        raise HTTPException(
            status_code=503,
            detail="System is currently processing another document. Please try again later.",
        )

    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Save uploaded file
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        # Process document
        chunks_created = await qa_system.process_document(content, file.filename)

        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.post("/question/stream", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded documents"""

    # Check if we have any documents (either in state or in Pinecone)
    pinecone_stats = qa_system.get_pinecone_stats()
    has_vectors = pinecone_stats and pinecone_stats.get("total_vectors", 0) > 0

    if not qa_system.current_documents and not has_vectors:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Please upload a document first.",
        )

    if qa_system.is_processing:
        raise HTTPException(
            status_code=503,
            detail="System is currently processing a document. Please try again in a moment.",
        )

    try:
        result = await qa_system.ask_question(request.question, request.top_k)

        return QuestionResponse(
            status="success",
            message="Answer generated successfully",
            question=request.question,
            answer=result["answer"],
            relevant_chunks=result["relevant_chunks"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store and reset state"""
    try:
        success = await qa_system.clear_all_data()

        if success:
            return JSONResponse(
                content={
                    "message": "All data cleared successfully",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to clear data. Check logs for details."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pinecone_stats = qa_system.get_pinecone_stats()
    return {
        "status": "healthy",
        "services": {
            "pinecone": qa_system.vector_store is not None,
            "openai": qa_system.model is not None,
            "retriever": qa_system.retriever is not None,
            "chain": qa_system.chain is not None,
        },
        "data": {
            "documents": len(qa_system.current_documents),
            "total_chunks": qa_system.total_chunks,
            "pinecone_vectors": (
                pinecone_stats.get("total_vectors", 0) if pinecone_stats else 0
            ),
        },
    }


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    return {
        "documents": qa_system.current_documents,
        "total_documents": len(qa_system.current_documents),
        "total_chunks": qa_system.total_chunks,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
