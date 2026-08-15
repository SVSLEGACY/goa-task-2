import os
import nltk
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# Configuration
DATASET_REPO = "ai4bharat/MSMARCO-XI"
FILENAME = "train/hintrain.parquet" # Contains both English and Hindi
MAX_ROWS = 500 # Just 500 for testing, change to 500k later
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "msmarco_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 

def chunk_text(text, child_target_tokens=128):
    """
    Splits text into sentences, groups them into child chunks.
    For simplicity, we estimate 1 token ~ 4 characters.
    """
    sentences = sent_tokenize(str(text))
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence) / 4
        if current_length + sentence_len > child_target_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def build_index():
    print(f"Downloading {FILENAME} from HF...")
    file_path = hf_hub_download(repo_id=DATASET_REPO, filename=FILENAME, repo_type="dataset")
    print(f"Loading Parquet file with pandas...")
    df = pd.read_parquet(file_path)
    
    # Filter short answers (checking both English and Hindi)
    df = df[(df['Answer'].str.len() >= 50) & (df['Eng_Answer'].str.len() >= 50)]
    
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_size = model.get_embedding_dimension()

    print(f"Initializing Qdrant client at {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)
    
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points = []
    processed_rows = 0
    
    # Collect all chunks first for batch encoding
    all_chunks = []
    all_metadata = []
    
    print("Collecting chunks...")
    for _, row in df.iterrows():
        if processed_rows >= MAX_ROWS:
            break
            
        passage_id = str(row.get("query_id", uuid.uuid4()))
        
        # We index BOTH English and Hindi answers
        langs_to_process = [
            ("hi", row.get("Answer")),
            ("en", row.get("Eng_Answer"))
        ]
        
        for lang, answer in langs_to_process:
            if not answer:
                continue
                
            parent_context = answer
            child_chunks = chunk_text(parent_context)
            
            for chunk in child_chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "passage_id": passage_id,
                    "language": lang,
                    "child_chunk": chunk,
                    "parent_context": parent_context
                })
                
        processed_rows += 1

    # Batch encode all chunks at once (10-50x faster than one-by-one)
    print(f"Batch encoding {len(all_chunks)} chunks...")
    all_vectors = model.encode(all_chunks, batch_size=64, show_progress_bar=True).tolist()
    
    # Create points
    print("Creating index points...")
    for i, (vector, metadata) in enumerate(zip(all_vectors, all_metadata)):
        points.append(
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=metadata)
        )
        
        # Batch insert every 1000 points
        if len(points) >= 1000:
            print(f"Inserting {len(points)} chunks ({i+1}/{len(all_chunks)})...")
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    if points:
        print(f"Inserting final {len(points)} chunks.")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print(f"Finished processing {processed_rows} rows (both English and Hindi).")
    print("Qdrant index is ready!")

if __name__ == "__main__":
    build_index()
