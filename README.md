# TrashAway Robot
This repository is divided into three different branches containing all the necessary material to:

<ul>
<li> -main:  make a PiCar-X robot perform the task of cleaning a squared environment from colored cubes </li>
<li> -RL: train the agent on a Coppeliasim simulated environment (already present, can be modified to perform other tasks) </li>
<li> -object_detection: identify objects in the environment through color detection </li>
</ul>

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
`./train.sh`

the training parameters can be changed in the script
# How to evaluate a configuration
Run the command below from the RL branch main directory
`python evaluate.py -w <your_weights.pt>`

# How to run the task in the real world
From the main branch main directory run the command:
`python server.py -parameters <your_weights.pt>`

then connect to the robot via ssh and run on it the following command:

`python client.py -host_address <server_IP_address>`
