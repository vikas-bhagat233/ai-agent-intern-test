import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))
    
    # Document precedence weights
    AUTHORITY_WEIGHT = {
        "current": 1.0,
        "superseded": 0.1,
        "legacy": 0.05,
        "internal": 0.0,  # Should never be returned to user
        "product": 0.8,
        "policy": 1.0
    }
    
    # Sensitive fields to never expose
    SENSITIVE_FIELDS = [
        "email", "address", "internal_notes", "risk_score", 
        "customer_id", "payment_method", "internal"
    ]
    
    # Order status priority (for determining current authoritative status)
    ORDER_STATUS_PRIORITY = {
        "cancelled": 0,
        "returned": 1,
        "delivered": 2,
        "shipped": 3,
        "processing": 4,
        "pending": 5
    }

config = Config()