#!/bin/bash
#SBATCH --job-name=af3_smoke_msa
#SBATCH --account=g205
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=/iopsstor/scratch/cscs/sreddy/af3_smoketest/logs/af3_smoke_msa_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sreddy/af3_smoketest/logs/af3_smoke_msa_%j.err

set -euo pipefail
echo "=== AF3 smoketest WITH MSA: MCC950 + NLRP3 ==="
echo "Start:   $(date)"
echo "Node:    $(hostname)"
echo ""

srun -A g205 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --environment=alphafold3_dynamic \
    python /root/run_alphafold.py \
    --json_path=/root/af_input/mcc950_nlrp3_with_msa.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output \
    --flash_attention_implementation=xla \
    --db_dir=/root/public_databases

echo ""
echo "End:     $(date)"
echo ""
echo "=== Output ==="
ls -la /iopsstor/scratch/cscs/sreddy/af3_smoketest/output/mcc950_nlrp3_with_msa/

echo ""
echo "=== Summary confidences ==="
cat /iopsstor/scratch/cscs/sreddy/af3_smoketest/output/mcc950_nlrp3_with_msa/mcc950_nlrp3_with_msa_summary_confidences.json 2>/dev/null
