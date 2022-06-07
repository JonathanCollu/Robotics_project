# TrashAway Robot
This repository contains all the necessary material to train a PiCar-X to perform the task of "cleaning" a squared environment from cubes. The training of the agent is performed using Deep Reinforcement Learning on a simulated CoppeliaSim environment.

## Authors
<a href="https://github.com/JonathanCollu">Jonathan Collu</a>, <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>, <a href="https://github.com/IrinaMonaEpure">Irina Mona Epure</a>, <a href="https://github.com/diegobc11">Diego Barreiro</a> and <a href="https://github.com/jorgie007">Ayodele Adetunji</a>

# Requirements
 To run these scripts, a `Python 3` environment is required, together with the necessary packages specified in the `requirements.txt` file. In order to install the requirements, run the following command from the main directory:
 
 ```
 pip install -r requirements.txt
 ````

# How to upload files on the PiCar-X robot
Run the following command from the main directory
```
./upload_file.sh <IP address of the picar> <local filepath>
```

# How to train an agent
For the training on the simulation it is necessary to copy the following files in the CoppeliaSim `src` directory: `Model.py`, `Reinforce.py`, `agent.py`, `color_detection.py` and `env.py`. The files `start.py` and `run.py` must be copied in the CoppeliaSim main folder.
Run the command below from the main directory
```
python run.py -run_name <run_name> -cp_name <checkpoint_name> -epochs <epochs_num> -M <traces_per_epoch> -T <trace_len> -gamma <discount>
```
An example is shown in `train.sh`.

# How to evaluate a trained agent
Run the command below from the main directory
```
python evaluate.py -w <your_weights.pt>
```

# How to run the task in the real world
From the main directory run the command:
```
python server.py -parameters <your_weights.pt>`
```
then connect to the robot via ssh and run on it the following command:
```
python client.py -host_address <server_IP_address>`
```

# Short demo on the simulated environment
<img src="https://github.com/riccardomajellaro/TrashAway_Robot/blob/main/sim_short_demo.gif" width="400" height="400"/>
