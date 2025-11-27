import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.core.providers import get_embeddings, get_vectordb, get_visual_search

async def main():
    print("Verifying Cloud Search...")
    
    # 1. Verify Text Search
    print("\n--- Text Search (AssetManual) ---")
    embeddings = get_embeddings()
    vectordb = get_vectordb()
    
    query = "how to reset the alarm"
    vector = await embeddings.embed_query(query)
    results = await vectordb.hybrid_search("AssetManual", query, vector, limit=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results")
    for res in results:
        props = res.get("properties", {})
        print(f"- [Page {props.get('page_number')}] {props.get('content')[:100]}...")

    # 2. Verify Visual Search
    print("\n--- Visual Search (PDFDocuments) ---")
    visual_search = get_visual_search()
    
    v_results = await visual_search.search(query="wiring diagram", top_k=3)
    print(f"Query: wiring diagram")
    print(f"Found {len(v_results)} results")
    for res in v_results:
        print(f"- [Page {res.page_number}] Score: {res.score:.3f} (Manual: {res.asset_manual})")

if __name__ == "__main__":
    asyncio.run(main())

