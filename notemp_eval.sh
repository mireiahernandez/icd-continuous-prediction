#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=mh1022@ic.ac.uk # required to send email notifcations - please replace <your_username> with your college login name or email address

source /vol/bitbucket/mh1022/dl_cw_pyenv/bin/activate
python main.py --num_chunks 4 --run_name "2206_1648_All_chunk_Run_temporal_4"
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
