import importlib, json
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
print("qwen4_exp 등록:", "qwen4_exp" in M, "->", M.get("qwen4_exp"))
for mod in ["transformers.models.qwen4_exp.modeling_qwen4_exp"]:
    try:
        m=importlib.import_module(mod); print("import OK:", m.__file__)
        names=[n for n in dir(m) if "PLE" in n.upper() or "Ngram" in n or "NGram" in n]
        print("PLE 관련 클래스:", names)
    except Exception as e: print("import 실패:", type(e).__name__, str(e)[:200])
try:
    import compressed_tensors; print("compressed_tensors", compressed_tensors.__version__)
except Exception as e: print("compressed_tensors 없음:", e)
c=json.load(open("/model/config.json")); q=c["quantization_config"]
print("quant:", {k:q[k] for k in q if k!="ignore"} if isinstance(q,dict) else q)
