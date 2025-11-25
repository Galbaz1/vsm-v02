#!/usr/bin/env python
"""
generate_previews.py

Usage:
    python generate_previews.py [pdf_path] [output_dir]

Generates PNG preview images for each page of a PDF manual.
Uses pdf2image if available, otherwise falls back to pdftoppm.
"""

import sys
import subprocess
from pathlib import Path


def generate_with_pdf2image(pdf_path: Path, output_dir: Path) -> bool:
    """Try to generate previews using pdf2image (Python library)."""
    try:
        from pdf2image import convert_from_path
        
        print(f"[pdf2image] Converting {pdf_path}...")
        images = convert_from_path(pdf_path)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        for page_number, image in enumerate(images, start=1):
            output_path = output_dir / f"page-{page_number}.png"
            image.save(output_path, "PNG")
            print(f"  Saved page {page_number} -> {output_path}")
        
        print(f"[pdf2image] Generated {len(images)} preview images.")
        return True
    except ImportError:
        print("[pdf2image] Not available, trying pdftoppm...")
        return False
    except Exception as e:
        print(f"[pdf2image] Error: {e}")
        return False


def generate_with_pdftoppm(pdf_path: Path, output_dir: Path) -> bool:
    """Try to generate previews using pdftoppm (system command)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / "page"
        
        print(f"[pdftoppm] Converting {pdf_path}...")
        result = subprocess.run(
            ["pdftoppm", "-png", str(pdf_path), str(prefix)],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # pdftoppm creates page-1.png, page-2.png, etc.
        # Count how many were created
        png_files = sorted(output_dir.glob("page-*.png"))
        print(f"[pdftoppm] Generated {len(png_files)} preview images.")
        return True
    except FileNotFoundError:
        print("[pdftoppm] Not found. Install poppler-utils or pdf2image.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[pdftoppm] Error: {e.stderr}")
        return False


def main():
    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path("static/manuals/uk_firmware_manual.pdf")
    
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        # Derive output dir from PDF name
        manual_name = pdf_path.stem.replace("_manual", "")
        output_dir = Path("static/previews") / manual_name
    
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)
    
    # Try pdf2image first, then pdftoppm
    if not generate_with_pdf2image(pdf_path, output_dir):
        if not generate_with_pdftoppm(pdf_path, output_dir):
            print("\nError: Could not generate previews.")
            print("Install one of:")
            print("  - pdf2image: pip install pdf2image")
            print("  - poppler-utils: brew install poppler (macOS) or apt-get install poppler-utils (Linux)")
            sys.exit(1)


if __name__ == "__main__":
    main()

