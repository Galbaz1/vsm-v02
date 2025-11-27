import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.core.providers import get_vectordb, get_visual_search

async def main():
    print("RESETTING CLOUD WEAVIATE...")
    print("This will delete 'AssetManual' and 'PDFDocuments' collections.")
    
    # Use WeaviateCloud provider to get client (it handles auth)
    # We need to access the underlying client to delete collections
    try:
        vectordb = get_vectordb()
        client = vectordb.connect()
        
        # Delete AssetManual
        if client.collections.exists("AssetManual"):
            client.collections.delete("AssetManual")
            print("Deleted 'AssetManual' collection.")
        else:
            print("'AssetManual' not found.")
            
        # Delete PDFDocuments
        if client.collections.exists("PDFDocuments"):
            client.collections.delete("PDFDocuments")
            print("Deleted 'PDFDocuments' collection.")
        else:
            print("'PDFDocuments' not found.")
            
        print("\nReset complete. Ready for fresh ingestion.")
        
    except Exception as e:
        print(f"Error during reset: {e}")

if __name__ == "__main__":
    asyncio.run(main())

