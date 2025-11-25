#!/usr/bin/env python
"""
weaviate_search_manual.py

Usage:
    python weaviate_search_manual.py "how do I reset the compressor alarm?"

- Runs a semantic (vector) search over the AssetManual collection.
- Prints top-k chunks with their content.
"""

import sys
import json
import weaviate


COLLECTION_NAME = "AssetManual"


def main():
    if len(sys.argv) < 2:
        print("Usage: python weaviate_search_manual.py \"your query here\"")
        sys.exit(1)

    query = sys.argv[1]

    with weaviate.connect_to_local() as client:
        coll = client.collections.use(COLLECTION_NAME)

        response = coll.query.near_text(
            query=query,
            limit=5,  # adjust as needed
        )

        print(f"=== Results for: {query!r} ===\n")
        for i, obj in enumerate(response.objects, start=1):
            props = obj.properties
            manual_name = props.get("manual_name", "")
            anchor_id = props.get("anchor_id", "")
            content = props.get("content", "")

            print(f"--- Result {i} ---")
            print(f"Manual: {manual_name}")
            print(f"Anchor: {anchor_id}")
            print()
            # Print first bit of content for inspection
            print(content[:800])
            if len(content) > 800:
                print("\n...[truncated]...\n")


if __name__ == "__main__":
    main()
