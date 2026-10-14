"""Environment verification script for ElderlyCare AI unified application."""

import sys

libs = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "streamlit",
    "easyocr",
    "pymupdf",
    "PIL",
    "reportlab"
]

print("Python version:", sys.version)
print("\nVerifying core dependencies for ElderlyCare AI...")
all_ok = True
for lib in libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [OK] {lib:<15} : {ver}")
    except ImportError as e:
        print(f"  [MISSING] {lib:<11} : {e}")
        all_ok = False

if all_ok:
    print("\n✅ All core dependencies for Medicine Verification and Fall Monitoring are installed!")
else:
    print("\n⚠️ Some dependencies are missing. Run: pip install -r requirements.txt")
