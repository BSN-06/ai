"""Quick import check — run with: venv\Scripts\python.exe check_imports.py"""
import sys

results = {}
packages = [
    ("fastapi", "fastapi"),
    ("sqlalchemy", "sqlalchemy"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic_settings"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("uvicorn", "uvicorn"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "Pillow"),
    ("sklearn", "scikit-learn"),
    ("httpx", "httpx"),
    ("pytest", "pytest"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
]

for module, pkg_name in packages:
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "?")
        results[pkg_name] = f"OK  ({ver})"
    except ImportError as e:
        results[pkg_name] = f"MISSING  ({e})"

print("\n=== PACKAGE IMPORT CHECK ===")
all_ok = True
for pkg, status in results.items():
    marker = "✓" if "OK" in status else "✗"
    print(f"  {marker} {pkg:<30} {status}")
    if "MISSING" in status and pkg not in ("torch", "ultralytics"):
        all_ok = False

print()
if all_ok:
    print("CORE IMPORTS: PASS (torch/ultralytics optional for core API)")
else:
    print("CORE IMPORTS: SOME PACKAGES MISSING")
sys.exit(0)
