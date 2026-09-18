from llama_index.core import VectorStoreIndex, Document
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import sys

load_dotenv()

def build_index(data_dir="data", filename="Sample_ICU_note.pdf"):
    filepath = f"{data_dir}/{filename}"

    if not os.path.exists(filepath):
        print(f"Error: no file found at '{filepath}'. Check the filename and that it's in the data folder.")
        sys.exit(1)

    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"Error: couldn't read the PDF. It may be corrupted or password-protected. Details: {e}")
        sys.exit(1)

    if not text.strip():
        print("Error: the PDF was read but no text was extracted. It may be a scanned image PDF that needs OCR.")
        sys.exit(1)

    documents = [Document(text=text)]

    try:
        index = VectorStoreIndex.from_documents(documents)
    except Exception as e:
        print(f"Error: couldn't build the index. This often means an issue with your OpenAI API key or credits. Details: {e}")
        sys.exit(1)

    return index, documents

def query_index(index, question):
    try:
        query_engine = index.as_query_engine(similarity_top_k=4)
        response = query_engine.query(question)
    except Exception as e:
        print(f"Error answering the question: {e}")
        return None

    print("\n--- Checkpoint 2: Retrieved chunks ---")
    for node in response.source_nodes:
        print(f"Score: {node.score:.3f}")
        print(node.text[:200])
        print("---")

    return response

if __name__ == "__main__":
    print("Loading and indexing document...")
    index, documents = build_index()

    print("\n--- Checkpoint 1: Document text ---")
    print(documents[0].text[:300])

    print("\nReady. Type your question (or 'quit' to exit):\n")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        response = query_index(index, question)
        if response is not None:
            print(f"\n--- Checkpoint 3: Final answer ---\n{response}\n")