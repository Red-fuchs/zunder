# tiny-random/flint -> 4-bit g32 with 8-bit embeddings (mirrors flint-int4's mixed bits), gate unquantised
import sys, json, shutil, os
from mlx_lm import load
from mlx_lm.utils import quantize_model, save_model, save_config
src, dst = sys.argv[1], sys.argv[2]
model, _ = load(src)
cfg = json.load(open(f"{src}/config.json"))
def pred(path, m):
    if path.endswith("mlp.gate"): return False
    if path.endswith("embed_tokens"): return {"bits": 8, "group_size": 32}
    return True
model, cfg = quantize_model(model, cfg, 32, 4, quant_predicate=pred)
os.makedirs(dst, exist_ok=True)
save_model(dst, model)
save_config(cfg, f"{dst}/config.json")
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]: shutil.copy(f"{src}/{f}", dst)
print({k: v for k, v in cfg["quantization"].items() if not isinstance(v, bool)} )
