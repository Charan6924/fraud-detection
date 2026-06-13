#!/bin/bash                                                                                                                                              
#SBATCH --job-name=fraud_grid_search                                                                                                                     
#SBATCH --partition=gpu                                                                                                                                  
#SBATCH --gres=gpu:1                                                                                                                                     
#SBATCH --cpus-per-task=6                                                                                                                                
#SBATCH --mem=64G                                                                                                                                        
#SBATCH --time=24:00:00                                                                                                                                  
#SBATCH --output=/home/cxv166/fraud-detection/logs/%j_grid_search.out                                                                                    
#SBATCH --error=/home/cxv166/fraud-detection/logs/%j_grid_search.err  

# Activate virtual environment if you have one
# source ~/venv/bin/activate
source /home/cxv166/fraud-detection/.venv/bin/activate

# Navigate to code directory
cd /home/cxv166/fraud-detection/ml/training/

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Run training
python feature_select.py

echo "End time: $(date)"
