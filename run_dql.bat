python start.py
timeout /t 5
python run.py ^
-run_name dql ^
-cp_name dql_20_weights ^
-epochs 1000 ^
-target_model ^
-tm_wait 25 ^
-rb_size 256 ^
-batch_size 64 ^
-policy egreedy ^
-epsilon 0.02 0.99 1000. ^
-gamma 0.9