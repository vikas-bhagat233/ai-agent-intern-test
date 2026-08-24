# Monkey-patch to resolve incompatibility between openai v1.12.0 and httpx >= 0.28.0
# where the proxies parameter was removed in httpx.Client.__init__
try:
    import openai._base_client
    from openai._base_client import SyncHttpxClientWrapper, AsyncHttpxClientWrapper

    class CustomSyncHttpxClientWrapper(SyncHttpxClientWrapper):
        def __init__(self, *args, **kwargs):
            kwargs.pop("proxies", None)
            super().__init__(*args, **kwargs)

    class CustomAsyncHttpxClientWrapper(AsyncHttpxClientWrapper):
        def __init__(self, *args, **kwargs):
            kwargs.pop("proxies", None)
            super().__init__(*args, **kwargs)

    openai._base_client.SyncHttpxClientWrapper = CustomSyncHttpxClientWrapper
    openai._base_client.AsyncHttpxClientWrapper = CustomAsyncHttpxClientWrapper
except Exception:
    pass

from .config import config
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .rag_agent import RAGAgent
from .order_tool import OrderTool
from .conversation import ConversationMemory

__version__ = "0.1.0"
__all__ = [
    'config',
    'DocumentProcessor',
    'VectorStore',
    'RAGAgent',
    'OrderTool',
    'ConversationMemory'
]