# TrashAway Robot
This repository contains all the necessary material to train a PiCar-X to perform the task of cleaning a squared environment from cubes.

## Authors
<a href="https://github.com/JonathanCollu">Jonathan Collu</a>, <a href="https://github.com/riccardomajellaro">Riccardo Majellaro</a>, <a href="https://github.com/IrinaMonaEpure">Irina Mona Epure</a>, <a href="https://github.com/diegobc11">Diego Barreiro</a> and <a href="https://github.com/jorgie007">Ayodele Adetunji</a>

# Requirements
 To run the available scripts and algorithm configurations, a `Python 3` environment is required, together with the required packages, specified in the `requirements.txt` file, in the main directory. In order to install the requirements, run the following command from the main directory: 
 
 ```
 pip install -r requirements.txt
 ````

# How to upload files on the PiCar-X robot

Run the following command from the main branch main directory

```
./upload_file.sh <IP address of the picar> <local filepath>
``` 

# How to train a configurations
For the training on the simulation is necessary to copy the files in this folder in the CoppeliaSim src directory. The files start.py and run.py must be copied in the CoppeliaSim main folder
`./train.sh`

the training parameters can be changed in the script
# How to evaluate a configuration
Run the command below from the main directory
`python evaluate.py -w <your_weights.pt>`

# How to run the task in the real world
From the main directory run the command:
`python server.py -parameters <your_weights.pt>`

then connect to the robot via ssh and run on it the following command:

`python client.py -host_address <server_IP_address>`
