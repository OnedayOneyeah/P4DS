import os
from typing import List
import pandas as pd
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain용 Embeddings 래퍼: sentence-transformers 모델을 사용."""
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
 
def ingest_pdf_to_faiss(
    pdf_path: str,
    index_dir: str = "faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    1. PDF 로드 → 페이지별 Document 생성
    2. 텍스트 청크 분할
    3. SentenceTransformer로 임베딩
    4. FAISS 인덱스 생성 및 저장
    """
    print(f"[1] PDF 로드 중: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    print(f"[2] 텍스트 청크 분할 중 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    print(f"[3] 임베딩 모델 준비 중...")
    embeddings = SentenceTransformerEmbeddings("BAAI/bge-m3")

    print(f"[4] FAISS 인덱스 생성 중...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"[5] 로컬 디렉토리에 인덱스 저장: {index_dir}")
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"\n✅ 완료: '{index_dir}'에 FAISS 인덱스 저장됨.")

def load_faiss_and_query(index_dir: str, query: str, top_k: int = 5):
    """저장된 FAISS 인덱스를 로드한 뒤 `query`로 검색해 결과를 출력 (테스트용)."""
    embeddings = SentenceTransformerEmbeddings("BAAI/bge-m3")
    vector_store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True  # ← 핵심 수정
    )

    docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)

    print(f"\n[검색 결과] \"{query}\" (top-{top_k})")
    print("=" * 80)
    for rank, (doc, score) in enumerate(docs_and_scores, 1):
        snippet = doc.page_content.replace("\n", " ")[:200] + "..."
        print(f"{rank:>2}. score={score:.4f} | {snippet}")
    print("=" * 80)

def get_faiss_results(index_dir: str, query: str, top_k: int = 5) -> dict:
    """FAISS 인덱스를 로드하여 검색 결과를 반환합니다."""
    embeddings = SentenceTransformerEmbeddings("BAAI/bge-m3")
    vector_store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)

    return {
        "documents": [doc.page_content for doc, _ in docs_and_scores],
        "scores": [score for _, score in docs_and_scores]
    }

def collection_query(query_texts, n_results, db_type="startup"):
    """
    db_type에 따라 startup(Chroma) 또는 stanford(FAISS)에서 유사 문서 검색.
    반환: dict - documents, scores (optional)
    """
    if db_type == "startup":
        # ChromaDB 검색
        DB_PATH = "./db/csv"
        client_chroma = chromadb.PersistentClient(path=DB_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ) 

        collection = client_chroma.get_collection(
            name="startup_collection",
            embedding_function=embedding_fn
        )

        results = collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

        return {
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "distances": results.get("distances", [])
        }

    elif db_type == "stanford":
        faiss_result = get_faiss_results(
            index_dir="./db/pdf",  # 인덱스 저장 경로
            query=query_texts[0],
            top_k=n_results
        )
        return {
            "documents": [faiss_result["documents"]],  # 통일된 리스트 형태
            "scores": faiss_result["scores"]
        }

    else:
        raise ValueError(f"Unknown db_type: {db_type}")

if __name__ == "__main__":
    PDF_PATH = "./db/hai_ai_index_report_2025.pdf"
    INDEX_DIR = "./db/pdf"

    # === PDF 파일 db 저장 ===
    if not os.path.isdir(INDEX_DIR):
        ingest_pdf_to_faiss(
            pdf_path=PDF_PATH,
            index_dir=INDEX_DIR,
            chunk_size=1200,
            chunk_overlap=200,
        )

    # === CSV 파일 db 저장 ===
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    CSV_PATH = "./db/new_DB.xlsx"
    DB_PATH="./db/csv"

    client_chroma = chromadb.PersistentClient(path=DB_PATH)
    local_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client_chroma.get_or_create_collection(
        name="startup_collection",
        embedding_function=local_embedding_fn
    )

    df = pd.read_excel(CSV_PATH)
    documents = []
    metadatas = []
    for idx, row in df.iterrows():
        description = str(row['설명']) if pd.notna(row['설명']) else ""
        proposal = str(row['제안서']) if pd.notna(row['제안서']) else ""
        summary = str(row['요약']) if pd.notna(row['요약']) else ""
        document_text = description + " " + summary

        metadata = row.to_dict()

        documents.append(document_text)
        metadatas.append(metadata)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(documents))]
    )

    print("✅ Saved to ChromaDB (using local embedding)")