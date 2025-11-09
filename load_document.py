import os
import sys
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_document_to_neo4j(file_path: str):
    """
    Load a document into Neo4j knowledge graph.
    
    Args:
        file_path: Path to document (.txt, .pdf, .docx)
    """
    logger.info(f"Starting document load: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        print(f"Error: File not found: {file_path}")
        return
    
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password or not neo4j_url:
        logger.error("Missing Neo4j credentials")
        print("Error: Set NEO4J_URI and NEO4J_PASSWORD in .env file")
        return
    
    file_ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"File type: {file_ext}")
    
    try:
        if file_ext == '.txt':
            loader = TextLoader(file_path)
        elif file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Error: Unsupported format: {file_ext}")
            print("Supported formats: .txt, .pdf, .docx")
            return
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        print(f"Loaded {len(documents)} pages")
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        print(f"Error loading document: {e}")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    print(f"Split into {len(chunks)} chunks")
    
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embeddings model loaded")
    
    print("Storing in Neo4j...")
    try:
        vector_store = Neo4jVector.from_documents(
            chunks,
            embeddings,
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_password,
            index_name="document_chunks",
            node_label="DocumentChunk",
            text_node_property="text",
            embedding_node_property="embedding",
        )
        
        logger.info(f"Successfully stored {len(chunks)} chunks")
        print("\n" + "="*70)
        print("Document loaded successfully")
        print("="*70)
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Chunks: {len(chunks)}")
        print("Search type: Vector + BM25 hybrid")
        print("\nRun: python enhanced_agent_clean.py")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Neo4j storage error: {e}")
        print(f"Error storing in Neo4j: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("Load Document to Neo4j")
        print("="*70)
        print("\nUsage:")
        print("  python load_document.py <file_path>")
        print("\nExamples:")
        print("  python load_document.py handbook.pdf")
        print("  python load_document.py notes.txt")
        print("  python load_document.py manual.docx")
        print("\nSupported formats: .txt, .pdf, .docx")
        print("="*70 + "\n")
    else:
        file_path = sys.argv[1]
        load_document_to_neo4j(file_path)