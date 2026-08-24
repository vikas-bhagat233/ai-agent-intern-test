import os
import logging
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import openai
from .config import config
import hashlib

class VectorStore:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "aster_row_docs"
        self.collection = None
        self._initialize_collection()
        
    def _initialize_collection(self):
        """Initialize or get the collection"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return random embedding for fallback (in production, better handling)
            return [0.0] * 1536
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            for chunk in doc['chunks']:
                chunk_id = hashlib.md5(
                    f"{doc['id']}_{chunk['text'][:100]}".encode()
                ).hexdigest()
                
                ids.append(chunk_id)
                documents_text.append(chunk['text'])
                
                # Prepare metadata - filter out sensitive info
                metadata = {
                    'filename': chunk['metadata'].get('filename', ''),
                    'doc_type': chunk['metadata'].get('doc_type', ''),
                    'authority_score': chunk['metadata'].get('authority_score', 0.5),
                    'headers': ' > '.join(chunk['headers'])[:200],
                    'source': chunk['metadata'].get('filename', '')
                }
                metadatas.append(metadata)
                
                # Get embedding
                embedding = self.get_embedding(chunk['text'])
                embeddings.append(embedding)
        
        # Add to collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
    
    def search(self, query: str, n_results: int = 5, 
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.get_embedding(query)
        
        # Build filter
        where_filter = {}
        if filter_metadata:
            for key, value in filter_metadata.items():
                where_filter[key] = value
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Get more for filtering
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process and filter results
            processed_results = []
            
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Filter out internal documents
                    if results['metadatas'][0][i].get('doc_type') == 'internal':
                        continue
                        
                    # Check authority threshold
                    authority = results['metadatas'][0][i].get('authority_score', 0)
                    if authority < 0.1:
                        continue
                    
                    processed_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'id': results['ids'][0][i]
                    })
            
            # Sort by authority * similarity
            processed_results.sort(
                key=lambda x: x['metadata'].get('authority_score', 0.5) * x['score'],
                reverse=True
            )
            
            return processed_results[:n_results]
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def clear(self):
        """Clear the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
        except:
            pass
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except:
            return {"total_chunks": 0, "collection_name": self.collection_name}