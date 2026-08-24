from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.session_id = None
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.history.append(message)
        self.last_updated = datetime.now()
        
    def get_messages(self, n_recent: Optional[int] = None) -> List[Dict]:
        """Get recent messages"""
        if n_recent:
            return list(self.history)[-n_recent:]
        return list(self.history)
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message"""
        for msg in reversed(self.history):
            if msg['role'] == 'user':
                return msg['content']
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message"""
        for msg in reversed(self.history):
            if msg['role'] == 'assistant':
                return msg['content']
        return None
    
    def get_context_window(self, max_tokens: int = 2000) -> str:
        """Get conversation context for prompt (simplified token counting)"""
        context = []
        for msg in self.history:
            context.append(f"{msg['role'].title()}: {msg['content']}")
        return "\n".join(context)
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        self.last_updated = datetime.now()
        
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            'total_messages': len(self.history),
            'user_messages': sum(1 for m in self.history if m['role'] == 'user'),
            'assistant_messages': sum(1 for m in self.history if m['role'] == 'assistant'),
            'duration': (datetime.now() - self.created_at).total_seconds(),
            'last_updated': self.last_updated.isoformat()
        }
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'history': list(self.history),
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'max_turns': self.max_turns
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMemory':
        """Create from dictionary"""
        memory = cls(max_turns=data.get('max_turns', 10))
        memory.session_id = data.get('session_id')
        memory.created_at = datetime.fromisoformat(data.get('created_at'))
        memory.last_updated = datetime.fromisoformat(data.get('last_updated'))
        for msg in data.get('history', []):
            memory.history.append(msg)
        return memory