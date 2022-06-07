python start.py
sleep 5
python run.py \
-run_name  \ # choose the name of the weights to be saved  
-cp_name  \ # load weights (if None train from random initialization)
-epochs 1000 \
-M 1 \
-T 7 \
-gamma 0.9
