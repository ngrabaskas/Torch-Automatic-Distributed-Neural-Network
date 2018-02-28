# Sample Code
To run use train.sh
Inside the shell script replace the following variables with their path on your system.

 - mpirun_path (replace with OpenMPI mpirun path)
 - hostfile_path (replace with path to hostfile)
 - luajit_path (replace with path to luajit)
 - train_path (replace with path to  sgd-torchad_nn-cifar.lua)


## Data on nodes
All nodes must have the example code in the exact same location. When OpenMPI is executed it will look in the same location on each node to find sgd-torchad_nn-cifar.lua.