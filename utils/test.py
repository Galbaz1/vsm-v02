import weaviate
from weaviate.classes.generate import GenerativeConfig

def main():
    # This ensures the socket is closed cleanly
    with weaviate.connect_to_local() as client:
        movies = client.collections.use("Movie")

        # Example: RAG with local Ollama model (pick your model)
        response = movies.generate.near_text(
            query="What is Weaviate?",
            limit=1,
            grouped_task="Answer in one short sentence.",
            generative_provider=GenerativeConfig.ollama(
                api_endpoint="http://localhost:11434",
                model="qwen2.5:7b",   # or "qwen3-vl:8b" or whatever you want
            ),
        )
        print(response.generative)  # or .text depending on client version

if __name__ == "__main__":
    main()
