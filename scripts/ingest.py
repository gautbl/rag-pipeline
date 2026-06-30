# scripts/ingest.py — Chargement et découpage des PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import os

def split_documents(pdf_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Charge tous les PDFs d'un répertoire et les découpe en chunks.
    Retourne une liste de chunks (LangChain Documents).
    """
    
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf")

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {pdf_dir}")

    print(f"📂 {len(pdf_files)} PDF(s) trouvé(s) dans {pdf_dir}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  📄 Chargement de {os.path.basename(pdf_path)}...")
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()
        print(f"      -> {len(docs)} page(s) chargée(s)")

        chunks = splitter.split_documents(docs)
        print(f"      -> {len(chunks)} chunks créés")

        all_chunks.extend(chunks)

    print(f"\n✅ Total : {len(all_chunks)} chunks générés depuis {len(pdf_files)} PDF(s)")
    return all_chunks