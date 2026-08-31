from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as C
print("config keys with qwen:", [k for k in M if "qwen" in k])
print("causal-lm keys with qwen:", [k for k in C if "qwen" in k])
import transformers, os
p=os.path.dirname(transformers.__file__)+"/models"
print("qwen dirs:", sorted(d for d in os.listdir(p) if "qwen" in d))
