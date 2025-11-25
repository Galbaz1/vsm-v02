"""FastAPI server for Multimodal RAG"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from api import rag as rag_api

app = FastAPI(
    title="Multimodal PDF RAG API",
    description="Query PDF documents using ColQwen2.5 and Qwen2.5-VL",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    text: str
    top_k: int = 3


class RetrievedPage(BaseModel):
    page_id: int
    asset_manual: str
    page_number: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_pages: list[RetrievedPage]


class RetrieveResponse(BaseModel):
    retrieved_pages: list[RetrievedPage]


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "healthy", "message": "RAG API is running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return rag_api.query(request.text, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: QueryRequest):
    try:
        pages = rag_api.retrieve_only(request.text, request.top_k)
        return {"retrieved_pages": pages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://localhost:8002")
    print("Docs: http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8002)
