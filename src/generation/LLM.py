"""Construct the Groq chat model; credentials are never stored in code."""
import os

from src.Config.settings import NateqSettings

def make_llm(settings: NateqSettings):
    """Load GROQ_API_KEY from environment/.env; NEVER hardcode credentials."""
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("Missing GROQ_API_KEY. Add it to your private .env")
    return ChatGroq(model=settings.groq_model_name, temperature=0)
