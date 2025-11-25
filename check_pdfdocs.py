#!/usr/bin/env python3
"""
Check PDFDocuments collection page counts
"""
import weaviate

def main():
    with weaviate.connect_to_local() as client:
        # Check if PDFDocuments collection exists
        if not client.collections.exists("PDFDocuments"):
            print("❌ PDFDocuments collection does not exist")
            print("   Run: python scripts/colqwen_ingest.py \"Technical Manual\"")
            print("   Run: python scripts/colqwen_ingest.py \"UK Firmware Manual\"")
            return
        
        collection = client.collections.get("PDFDocuments")
        
        # Get total count
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"✓ Total pages in PDFDocuments: {total}\n")
        
        # Count by manual
        tech_manual_result = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("asset_manual").equal("Technical Manual"),
            limit=1000
        )
        
        uk_firmware_result = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("asset_manual").equal("UK Firmware Manual"),
            limit=1000
        )
        
        tech_count = len(tech_manual_result.objects) if tech_manual_result.objects else 0
        uk_count = len(uk_firmware_result.objects) if uk_firmware_result.objects else 0
        
        print("📚 Manuals in PDFDocuments collection:")
        if tech_count > 0:
            print(f"  - Technical Manual: {tech_count} pages")
        else:
            print("  ⚠️  Technical Manual: NOT INGESTED")
            
        if uk_count > 0:
            print(f"  - UK Firmware Manual: {uk_count} pages")
        else:
            print("  ⚠️  UK Firmware Manual: NOT INGESTED")
        
        print("\n📊 Expected page counts:")
        print("  - Technical Manual: 132 pages (techman.pdf)")
        print("  - UK Firmware Manual: 128 pages (uk_firmware.pdf)")
        
        # Verify against expected
        if tech_count == 132:
            print("\n✅ Technical Manual: CORRECT (132 pages)")
        elif tech_count > 0:
            print(f"\n⚠️  Technical Manual: MISMATCH (expected 132, got {tech_count})")
        
        if uk_count == 128:
            print("✅ UK Firmware Manual: CORRECT (128 pages)")
        elif uk_count > 0:
            print(f"⚠️  UK Firmware Manual: MISMATCH (expected 128, got {uk_count})")

if __name__ == "__main__":
    main()
