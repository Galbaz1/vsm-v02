#!/usr/bin/env python
"""
Quick test script to verify the FastAPI search API and static file serving.
"""

import requests

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint."""
    resp = requests.get(f"{BASE_URL}/healthz")
    print(f"Health check: {resp.json()}")
    assert resp.status_code == 200

def test_search():
    """Test search endpoint."""
    resp = requests.get(f"{BASE_URL}/search", params={"query": "battery test", "limit": 3})
    print(f"\nSearch results:")
    data = resp.json()
    print(f"Query: {data['query']}")
    print(f"Found {len(data['hits'])} hits\n")
    
    for i, hit in enumerate(data['hits'], 1):
        print(f"Hit {i}:")
        print(f"  Page: {hit['page_number']}")
        print(f"  PDF URL: {hit['pdf_page_url']}")
        print(f"  Preview URL: {hit['page_image_url']}")
        print(f"  Content preview: {hit['content'][:100]}...")
        print()

def test_static_files():
    """Test static file serving."""
    # Test PDF
    pdf_resp = requests.get(f"{BASE_URL}/static/manuals/uk_firmware_manual.pdf", stream=True)
    print(f"\nPDF file: {pdf_resp.status_code} (Content-Type: {pdf_resp.headers.get('Content-Type')})")
    
    # Test preview image
    img_resp = requests.get(f"{BASE_URL}/static/previews/uk_firmware/page-12.png")
    print(f"Preview image: {img_resp.status_code} (Content-Type: {img_resp.headers.get('Content-Type')})")
    
    assert pdf_resp.status_code == 200
    assert img_resp.status_code == 200

if __name__ == "__main__":
    print("Testing FastAPI Manual Search API...\n")
    try:
        test_health()
        test_search()
        test_static_files()
        print("\n✅ All tests passed!")
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: API server not running.")
        print("Start it with: uvicorn api.main:app --reload")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")

