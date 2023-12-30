#!/bin/bash
#SBATCH --account=pi-haihaolu

# for i in {1..13}
# do
#     sbatch run.sh-$i
# done

echo "Job started at $(date)"


sbatch /home/ymeng3/script/run.sh-36
