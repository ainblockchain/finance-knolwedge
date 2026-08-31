import inspect
try:
    import fla; print("fla", fla.__version__)
except Exception as e: print("fla 없음:", type(e).__name__, str(e)[:100])
import transformers; print("transformers", transformers.__version__)
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
print("qwen4_exp import OK; fla 사용 여부 심볼:", [n for n in dir(M) if 'fla' in n.lower() or 'chunk_gated' in n][:8])
from compressed_tensors.compressors.quantized_compressors import pack_quantized as pq
print("pack_quantized 심볼:", [n for n in dir(pq) if 'pack' in n.lower()])
print(inspect.getsource(pq.unpack_from_int32))
