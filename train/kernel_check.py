import importlib
for m in ["fla","flash_attn","causal_conv1d","flashinfer","triton","accelerate","peft"]:
    try:
        mod=importlib.import_module(m); print(f"{m:14s} OK", getattr(mod,'__version__','?'))
    except Exception as e: print(f"{m:14s} 없음 ({type(e).__name__})")
try:
    from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer as Q
    import inspect; src=inspect.getsource(Q)
    print("run_compressed 지원:", "run_compressed" in src)
    from compressed_tensors.linear.compressed_linear import CompressedLinear; print("CompressedLinear OK")
except Exception as e: print("quantizer 확인 실패:", type(e).__name__, str(e)[:150])
