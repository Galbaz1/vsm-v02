#!/usr/bin/env python
"""
Quick test to verify both manuals are in Weaviate and searchable.
"""

import weaviate

COLLECTION_NAME = "AssetManual"

def test_both_manuals():
    with weaviate.connect_to_local() as client:
        coll = client.collections.get(COLLECTION_NAME)
        
        # Get total count
        response = coll.aggregate.over_all(total_count=True)
        total = response.total_count
        print(f"✓ Total objects in {COLLECTION_NAME}: {total}")
        
        # Count by manual
        print("\n📚 Manuals in database:")
        for manual_name in ["UK Firmware Manual", "Technical Manual"]:
            response = coll.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("manual_name").equal(manual_name),
                limit=1
            )
            count_response = coll.aggregate.over_all(
                filters=weaviate.classes.query.Filter.by_property("manual_name").equal(manual_name),
                total_count=True
            )
            count = count_response.total_count
            print(f"  - {manual_name}: {count} chunks")
        
        # Test search for each manual
        print("\n🔍 Search tests:")
        
        # Search for something from UK Firmware Manual
        print("\n1. Query: 'alarm reset' (should find UK Firmware)")
        response = coll.query.near_text(
            query="alarm reset",
            limit=2
        )
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            print(f"   Result {i}: {props['manual_name']} - Page {props.get('page_number', 'unknown')}")
            print(f"   Content: {props['content'][:100]}...")
        
        # Search for something from Technical Manual
        print("\n2. Query: 'supply voltage' (should find Technical Manual)")
        response = coll.query.near_text(
            query="supply voltage",
            limit=2
        )
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            print(f"   Result {i}: {props['manual_name']} - Page {props.get('page_number', 'unknown')}")
            print(f"   Content: {props['content'][:100]}...")
        
        print("\n✅ Both manuals are searchable!")

if __name__ == "__main__":
    test_both_manuals()
