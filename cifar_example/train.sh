echo "enter number of nodes"
read nodes
echo "enter batchsize"
read batchsize
echo "enter # iterations"
read iter

#Setting paths
mpirun_path='/opt/openmpi-2.0.1/bin/mpirun'
hostfile_path='/home/ubuntu/TorchMPI-master/hostfile'
luajit_path='/home/ubuntu/torch/install/bin/luajit'
train_path='/home/ubuntu/testcase-demos/cifar_example/sgd-torchad_nn-cifar.lua'

$mpirun_path -n $nodes -npernode 1 --hostfile $hostfile_path --bind-to none $luajit_path $train_path -batchSize $batchsize -threads 1 -iterations $iter -learningRate 0.001 -data -usegpu    


