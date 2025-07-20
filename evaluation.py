# evaluation.py
from phoenix.trace.openai import OpenAIInstrumentor

# Automatically logs LLM requests
OpenAIInstrumentor().instrument()
