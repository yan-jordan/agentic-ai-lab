from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')

CHAT_MODEL = os.getenv('CHAT_MODEL')

SMALL_MODEL = os.getenv("SMALL_MODEL")

EMBED_MODEL = os.getenv('EMBED_MODEL')

