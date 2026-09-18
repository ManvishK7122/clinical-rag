from llama_index.core import VectorStoreIndex, Document
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def build_index(data_dir="data"):
    reader = PdfReader(f"{data_dir}/Sample_ICU_note.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    documents = [Document(text=text)]
    index = VectorStoreIndex.from_documents(documents)
    return index, documents

def query_index(index, question):
    query_engine = index.as_query_engine(similarity_top_k=4)
    response = query_engine.query(question)
    
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
        question = input("Question: ")
        if question.lower() == "quit":
            break
        response = query_index(index, question)
        print(f"\n--- Checkpoint 3: Final answer ---\n{response}\n")