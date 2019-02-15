# Traffic Orchestrator w/ LSTM & RL


## Executing TO (c++)

### Configuring TO 

change the recieve/send adress/port in the TO_Config.json file located in ./include.

### Run TO

- not optimised for CUDA. due to library dependancies.

- build docker image 

If image is directly built from the repository:

- download libtorch library here : https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip and replace with libtorch library in directory
  
- run docker image

- once in bash on the container `./runner.sh`


## Executing AI (python)

### Requirements 

### Running & Training LSTM 

- optimised for CUDA with Torch

- `python3 LSTM_Model.py`

- Hyper Parameters(iteration, learning rate) can be tuned in LSTM, uses US101 DataSet to predict the next sequence in the following timestep.

- Accuracy of Training and Accuracy of Test given on completion, with an lstm_model.pt file. This file can be used to test the model in the TO, by replacing the file inside ./include

### Running & Training RL 

- optimised for CUDA with Torch

- `python3 RL_Model.py`

- Again, Hyper paramters can be tuned. Uses Random Forest Classifier found in AI, as a means of assigning reward. 

- Random Forest Classifier Trains first and tested, both given and training for RL starts.

- MSE Given for RL, global optimum 0.0047, 

- rl_model.pt given as well to simulate the process in the TO, replace file in ./include with the new rl_model.pt 


## Dataset

- csv files parsed the data from US101 dataset. The file with the calculated heading is used for training of all the models. Only contains the instances in the data where there is a lane merge scenario and disregards all other driving scenarions. Presents the data in a time series fashion.
