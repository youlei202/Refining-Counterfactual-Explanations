##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash
### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J heloc
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
###BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:59
# request system-memory
#BSUB -R "rusage[mem=50GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u leiyo@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo data/logs/heloc_std.out
#BSUB -eo data/logs/heloc_std.err
# -- end of LSF options --


module load pandas/1.4.1-python-3.9.11 
module load scipy/1.7.3-python-3.9.11
module load python3/3.9.11
module load gurobipy/gurobi-9.5.2-python-3.9.11 

python3 -m runs.heloc