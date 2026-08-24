import os
import re
import frontmatter
from typing import List, Dict, Any
from pathlib import Path
import hashlib

class DocumentProcessor:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = []
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all markdown files with frontmatter from knowledge base"""
        self.documents = []
        
        for md_file in self.knowledge_base_path.glob("*.md"):
            doc = self._parse_markdown_file(md_file)
            if doc:
                self.documents.append(doc)
                
        return self.documents
    
    def _parse_markdown_file(self, filepath: Path) -> Dict[str, Any]:
        """Parse a single markdown file with frontmatter"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse frontmatter
            post = frontmatter.loads(content)
            metadata = post.metadata
            
            # Extract content without frontmatter
            content_text = post.content
            
            # Add file metadata
            metadata['filename'] = filepath.name
            metadata['filepath'] = str(filepath)
            
            # Determine document type and authority
            doc_type = self._classify_document(metadata, content_text)
            metadata['doc_type'] = doc_type
            metadata['authority_score'] = self._calculate_authority(metadata, doc_type)
            
            # Split into chunks if needed
            chunks = self._chunk_document(content_text, metadata)
            
            return {
                'metadata': metadata,
                'content': content_text,
                'chunks': chunks,
                'id': hashlib.md5(str(filepath).encode()).hexdigest()
            }
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def _classify_document(self, metadata: Dict, content: str) -> str:
        """Classify document type based on metadata and content"""
        status = metadata.get('status', '').lower()
        audience = metadata.get('audience', '').lower()
        authority = metadata.get('policy_authority', '').lower()
        answering = metadata.get('customer_answering')
        
        if (
            'internal' in status or 
            'internal' in audience or 
            authority == 'none' or 
            answering is False or
            'internal' in content.lower()[:200]
        ):
            return 'internal'
        if 'superseded' in status or 'legacy' in status:
            return 'superseded'
        if 'legacy' in metadata.get('version', '').lower():
            return 'legacy'
        if 'current' in status or 'active' in status:
            return 'current'
        if 'product' in metadata.get('category', '').lower() or 'product' in metadata.get('title', '').lower():
            return 'product'
        return 'policy'  # default
    
    def _calculate_authority(self, metadata: Dict, doc_type: str) -> float:
        """Calculate authority score for document"""
        from .config import config
        
        base_score = config.AUTHORITY_WEIGHT.get(doc_type, 0.5)
        
        # Boost for newer dates
        if 'date' in metadata:
            try:
                # Simple boost for newer docs - in production use proper date parsing
                if '2025' in metadata['date'] or '2024' in metadata['date']:
                    base_score *= 1.2
            except:
                pass
                
        # Penalize if explicitly marked as superseded
        if 'superseded_by' in metadata:
            base_score *= 0.1
            
        return min(base_score, 1.0)
    
    def _chunk_document(self, content: str, metadata: Dict, chunk_size: int = 500) -> List[Dict]:
        """Split document into semantic chunks"""
        chunks = []
        
        # Split by headers first
        sections = re.split(r'(#+\s+.*?\n)', content)
        
        current_chunk = ""
        current_headers = []
        
        for section in sections:
            if re.match(r'#+\s+', section):
                # This is a header
                current_headers.append(section.strip())
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'headers': current_headers.copy(),
                        'metadata': metadata.copy()
                    })
                    current_chunk = ""
            else:
                # This is content
                current_chunk += section
                
                if len(current_chunk) > chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'headers': current_headers.copy(),
                        'metadata': metadata.copy()
                    })
                    current_chunk = ""
        
        # Add remaining content
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'headers': current_headers.copy(),
                'metadata': metadata.copy()
            })
            
        return chunks
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of loaded documents"""
        if not self.documents:
            return {"total_documents": 0, "by_type": {}}
            
        by_type = {}
        for doc in self.documents:
            doc_type = doc['metadata'].get('doc_type', 'unknown')
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
            
        return {
            "total_documents": len(self.documents),
            "by_type": by_type,
            "documents": [{
                "filename": d['metadata'].get('filename'),
                "type": d['metadata'].get('doc_type'),
                "authority": d['metadata'].get('authority_score'),
                "chunks": len(d['chunks'])
            } for d in self.documents]
        }