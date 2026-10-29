# usage: run4787.sh <venv>; prints bench+prof+gate
V=$1; export HF_HUB_OFFLINE=1; cd ~/Projects/zunder/ref/dflash
$V/bin/python -I -c "import mlx.core as mx;print(mx.__version__)"
for i in 1 2 3; do $V/bin/python ../bench/qmm_bench.py; done
for sm in 1 0; do echo "== SMALLM=$sm prof"; for i in 1 2 3; do SMALLM=$sm $V/bin/python prof.py 2>&1 | tail -3; done
echo "== SMALLM=$sm gate"; SMALLM=$sm MODE=fast CHECK=1 REP=3 $V/bin/python gate_fast.py 2>&1 | tail -8; done
