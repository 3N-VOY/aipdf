from fastapi import FastAPI, UploadFile, HTTPException, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import tempfile
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uuid
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from pinecone import Pinecone, ServerlessSpec
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="PDF Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq LLM
llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = "pdf-search"
DIMENSION = 768

# Store the current active document namespace
CURRENT_NAMESPACE = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[str] = None

# Sanitize namespace name to avoid Pinecone errors
def sanitize_namespace(name):
    # Remove file extension
    name = name.replace('.pdf', '')
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Ensure it's not too long (Pinecone may have limits)
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized

# Ensure Pinecone index exists
def create_index_if_not_exists():
    try:
        pc.describe_index(PINECONE_INDEX_NAME)
        print(f"Found existing Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

# Check if namespace exists
def namespace_exists(index, namespace):
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return namespace in namespaces
    except Exception as e:
        print(f"Error checking namespace existence: {str(e)}")
        return False

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@app.on_event("startup")
async def startup_event():
    create_index_if_not_exists()
    print("API started and connected to Pinecone index")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Generate a unique namespace for this document
        global CURRENT_NAMESPACE
        CURRENT_NAMESPACE = sanitize_namespace(file.filename)

        print(f"Processing PDF: {file.filename} with namespace: {CURRENT_NAMESPACE}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process PDF
        docs = PyPDFLoader(temp_path).load()
        print(f"Loaded {len(docs)} pages from PDF")

        # Split text into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks from PDF")

        # Add enhanced metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["filename"] = file.filename
            # Ensure page metadata exists
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata.get("page", "unknown")

        # Connect to Pinecone index
        index = pc.Index(PINECONE_INDEX_NAME)

        # Only try to delete the namespace if it exists
        if namespace_exists(index, CURRENT_NAMESPACE):
            try:
                index.delete(delete_all=True, namespace=CURRENT_NAMESPACE)
                print(f"Cleared namespace {CURRENT_NAMESPACE} in Pinecone index")
            except Exception as e:
                # Log the error but continue processing
                print(f"Warning: Failed to delete namespace {CURRENT_NAMESPACE}: {str(e)}")
        else:
            print(f"Namespace {CURRENT_NAMESPACE} doesn't exist yet, skipping deletion")

        # Store in Pinecone with namespace
        vector_store = PineconeVectorStore(
            embedding=embeddings,
            index=index,
            namespace=CURRENT_NAMESPACE
        )

        ids = vector_store.add_documents(documents=chunks)
        print(f"Added {len(ids)} chunks to Pinecone namespace: {CURRENT_NAMESPACE}")

        # Cleanup
        os.unlink(temp_path)

        return {
            "message": f"PDF processed successfully. Created {len(chunks)} chunks in namespace {CURRENT_NAMESPACE}.",
            "namespace": CURRENT_NAMESPACE
        }

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        global CURRENT_NAMESPACE
        if not CURRENT_NAMESPACE:
            raise HTTPException(
                status_code=400,
                detail="No document has been uploaded yet. Please upload a PDF first."
            )

        print(f"Processing question: {request.question} in namespace: {CURRENT_NAMESPACE}")

        # Query vector store with the current namespace
        index = pc.Index(PINECONE_INDEX_NAME)
        vector_store = PineconeVectorStore(
            embedding=embeddings,
            index=index,
            namespace=CURRENT_NAMESPACE
        )

        # Retrieve relevant documents
        results = vector_store.similarity_search(request.question, k=5)
        # print(f"Retrieved {len(results)} chunks for question from namespace: {CURRENT_NAMESPACE}")


        if not results:
            return QuestionResponse(
                answer="I don't have enough information in the document to answer this question.",
                context="No relevant content found in the document."
            )

        # Format context with clear section markers
        context_parts = []
        for i, doc in enumerate(results):
            # Add document metadata to help LLM understand the source
            metadata_str = f"[Document: {doc.metadata.get('filename', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]"
            context_parts.append(f"DOCUMENT SECTION {i+1} {metadata_str}:\n{doc.page_content}")

        context = "\n\n" + "\n\n".join(context_parts)

        # Enhanced system prompt
        system_message = """
        You have access to a PDF document. Your task is to answer the user's questions strictly based on the content of the PDF. 
        If a question cannot be answered from the PDF, respond with: "The answer is not found in the document." 
        Be accurate, concise, and reference relevant sections or quotes when possible. Wait for the user's question.
        """

        # Improved prompt format
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"""CONTEXT:
{context}

QUESTION: {request.question}

Remember: ONLY use information from the document provided. If the answer isn't in the document, say "I don't have enough information in the document to answer this question."
""")
        ]

        print(f"Sending prompt to LLM with context length: {len(context)}")

        # Get answer from LLM
        response = llm.invoke(messages)
        print(f"Received response from LLM")

        return QuestionResponse(
            answer=response.content,
            context=context
        )

    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/index-info")
async def get_index_info():
    """Get information about the Pinecone index for debugging"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        return {
            "index_name": PINECONE_INDEX_NAME,
            "vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", DIMENSION),
            "namespaces": stats.get("namespaces", {}),
            "current_namespace": CURRENT_NAMESPACE
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/clear-index")
async def clear_index():
    """Clear the entire Pinecone index for debugging"""
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        global CURRENT_NAMESPACE
        CURRENT_NAMESPACE = None
        return {"message": "Index cleared successfully"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)