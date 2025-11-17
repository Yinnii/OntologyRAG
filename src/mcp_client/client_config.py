import os, openai, ollama

def get_client_model() -> tuple:
    client_type = os.getenv("CLIENT_TYPE", "openai")

    if client_type == "openai":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        return client, model
    elif client_type == "azure":
        client = openai.AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT")
        )
        model = os.getenv("AZURE_LLM_MODEL", "gpt-4o-mini")
        return client, model
    elif client_type == "litellm":
        client = openai.OpenAI(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL", "http://litellm.warhol.informatik.rwth-aachen.de")
        )

        model = os.getenv("LITE_LLM_MODEL", "gpt-4o-mini")
        return client, model
    elif client_type == "ollama":
        client = ollama.AsyncClient(
            host="http://warhol.informatik.rwth-aachen.de:11434/api"
        )
        model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        return client, model
    else:
        raise ValueError(f"Unsupported client type: {client_type}")  