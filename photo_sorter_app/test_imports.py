import sys
print(f"Python: {sys.version}")

try:
    import insightface
    print(f"insightface OK: {insightface.__version__}")
except Exception as e:
    print(f"insightface HATA: {e}")
    import traceback
    traceback.print_exc()

try:
    import onnxruntime
    print(f"onnxruntime OK: {onnxruntime.__version__}")
except Exception as e:
    print(f"onnxruntime HATA: {e}")
    import traceback
    traceback.print_exc()
