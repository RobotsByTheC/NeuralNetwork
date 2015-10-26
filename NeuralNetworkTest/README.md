# "Test" code for the neural network library

This code started out as a way to test the network code, but now also contains the robot heading training program. This is an Eclipse project.

### FileTest

Loads a neural network from a file and trains it, saving the output. 

### RobotHeadingTest

A failed attempt to make the neural network automatically teach itself how to rotate a simulated robot to a desired heading.

### HeadingNeuralNetworkTrainer

The code that trains the network based on recorded driver movements from the robot. There is code in the neural-network branch of [CMonster2015](https://github.com/RobotsByTheC/CMonster2015) that does this data recording.

The application loads a set of CSV files from the chosen directory and displays them on the graph. The data from all files in the directory are combined. Currently the eta (learning rate) and momentum training parameters can be adjusted. Pressing the "Train" button starts the training process.

In the `data/recorded-data` directory, there are some sample data directories. 
- `random-data`: multiple randomly generated data files.
- `cubic-data`: data for a cubic function. 
- `cubic-sparse-data`: the same cubic function, but with fewer data points
- `robot-data`: data recorded from the robot
