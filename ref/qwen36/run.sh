export HF_HUB_OFFLINE=1
cd ~/Projects/zunder/ref/qwen36
/opt/miniconda3/bin/python bench.py jang 2>&1 | grep -v Warning
~/.venvs/zunder-4787/bin/python -I bench.py flint 2>&1 | grep -v Warning
