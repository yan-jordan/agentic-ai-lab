from lab.settings import CHAT_MODEL , EMBED_MODEL , OLLAMA_BASE_URL , SMALL_MODEL
from langchain_ollama import ChatOllama , OllamaEmbeddings

def chat_model(model: str | None = None , temperature: float = 0.0 , **kwargs) -> ChatOllama:
    return ChatOllama(
    model = model or CHAT_MODEL,
    base_url = OLLAMA_BASE_URL,
    temperature = temperature,
        **kwargs
    )

def small_model(temperature: float = 0.0 , **kwargs) -> ChatOllama:
    return ChatOllama(
        model = SMALL_MODEL,
        base_url= OLLAMA_BASE_URL,
        temperature = temperature,
        **kwargs
    )

def embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model= EMBED_MODEL,
        base_url= OLLAMA_BASE_URL
    )