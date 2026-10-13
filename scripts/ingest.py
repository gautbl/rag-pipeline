# scripts/ingest.py — Chargement et découpage des PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import os


def split_documents(pdf_dir: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Charge tous les PDFs d'un répertoire et les découpe en chunks.

    Les PDFs corrompus ou vides sont ignorés avec un avertissement, sans
    interrompre le traitement des autres fichiers.

    Args:
        pdf_dir: Chemin du répertoire contenant les PDFs.
        chunk_size: Nombre de caractères par chunk.
        chunk_overlap: Chevauchement entre chunks consécutifs.

    Returns:
        Liste de chunks (LangChain Documents).

    Raises:
        FileNotFoundError: Si aucun fichier PDF n'est trouvé dans le répertoire.
    """

    pdf_files = [f for f in glob.glob(os.path.join(pdf_dir, "*"))
                 if f.lower().endswith(".pdf")]

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
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        except Exception as e:
            print(f"      ⚠️ Erreur sur {os.path.basename(pdf_path)} : {e} — fichier ignoré")
            continue

        if len(docs) == 0:
            print(f"      ⚠️ {os.path.basename(pdf_path)} : 0 pages — ignoré")
            continue

        print(f"      -> {len(docs)} page(s) chargée(s)")

        chunks = splitter.split_documents(docs)
        print(f"      -> {len(chunks)} chunks créés")

        all_chunks.extend(chunks)

    if all_chunks:
        print(f"\n✅ Total : {len(all_chunks)} chunks générés depuis {len(pdf_files)} PDF(s)")
    else:
        print(f"\n⚠️ Aucun chunk généré — tous les PDFs étaient vides ou corrompus.")
    return all_chunks