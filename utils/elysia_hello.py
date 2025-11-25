from elysia import configure, Tree

# Tell Elysia to use Ollama + your local models
configure(
    base_provider="ollama",
    complex_provider="ollama",
    base_model="qwen2.5:7b",   # decision / light work
    complex_model="qwen2.5:7b",  # you can swap to "qwen3-vl:8b" later
    model_api_base="http://localhost:11434",
)

tree = Tree()

print("=== base_lm sanity ===")
print(tree.base_lm("Say hi in one short sentence."))

print("\n=== full tree (with instructions) ===")
print(tree("Explain in 2 sentences what a vector database is."))
