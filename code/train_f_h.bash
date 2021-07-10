#！ /bin/sh
python 1_traj.py -option=0
python 2_f_trainer.py -option=0
python 3_h_trainer.py -option=0

python 1_traj.py -option=1
python 2_f_trainer.py -option=1
python 3_h_trainer.py -option=1

python 1_traj.py -option=2
python 2_f_trainer.py -option=2
python 3_h_trainer.py -option=2
