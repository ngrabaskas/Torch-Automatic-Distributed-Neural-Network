# Torch-Automatic-Distributed-Neural-Network
Torch Automatic Distributed Neural Network (TorchAD-NN) training library. Built on top of TorchMPI this module will automatically parallelize neural network training.

Main contribution is reducing the implementation complexity of data parallel neural network training by more than 90% and providing components, with near zero implementation complexity, to execute model parallel training on all or only select fully-connected neural layers.

See Thesis **insert link**

## Before Installation
Install [Torch 7.0](https://github.com/torch/torch7) and [TorchMPI](https://github.com/facebookresearch/TorchMPI)

**Note:** TorchMPI must be built using OpenMPI 2.0.1

Additional Dependencies

 - cutorch
 - cunn
 - torchnet

## Source Code
To install:
```bash
git clone https://github.com/ngrabaskas/Torch-Automatic-Distributed-Neural-Network.git
cd Torch-Automatic-Distributed-Neural-Network
luarocks make 
```
To remove:
```bash
luarocks remove torchad_nn 
```
## TorchMPI
If you wish to start MPI yourself, use the following commands. This is necessary to use the manual synchronize function.
```lua
mpi = require('torchmpi')
mpi.start(true)  --true equals use GPU
```
The mpi handle must be passed to the parallelize function and synchronizeModel function.
## To use Data Parallelism

Load TorchAD-NN library 
```lua
automation = require 'torchad_nn.datamodule'
```
Then after loading your data and before pre-processing call automated parallelization function. For example:
```lua
----------------------------------------------------------------------
-- Load Dataset Example
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
size = data.size()

----------------------------------------------------------------------
-- Call Automated Parallelization 
data, labels, size = automation.parallelize(data, labels, model, size, mpi, mpinn, batchSize) 

----------------------------------------------------------------------
-- preprocess/normalize data
...

```

This parallelize() function will split dataset evenly across all nodes in the MPI handle. Synchronization will occur automatically as long as stochasticgradient:train() or model:backward() is being used in training. Mpi and mpinn handles can be replaced with 'nil' and TorchAD-NN will automatically start them.

To set synchronization manually use:
```lua 
model:backward()

-- turn off automatic synchronize by using -1 
data, labels, size = automation.parallelize(data, labels, model, size, mpi, mpinn, -1) 

-- after backward propagation place synchronize call
automation.synchronizeModel(model, mpi)
```

## To use Model Parallelism

Load TorchAD-NN library 
```lua
require 'torchad_nn'
```
Start MPI
```lua
mpi = require('torchmpi')
mpi.start(true)  --true equals use GPU
```

Use new neural layer component names.
```lua 
-- old components
model:add(nn.Reshape(1024))
model:add(nn.Linear(1024, 2048))
model:add(nn.Tanh())
model:add(nn.Linear(2048,10))

-- new parallelized components
model:add(nn.MPInitialReshape(1024))
model:add(nn.MPInitialLinear(1024, 2048))
model:add(nn.MPTanh())
model:add(nn.MPBaseLinear(2048,10))
```
**Five available components:**
 - MPInitialReshape()*
 - MPInitialLinear()*
 - MPBaseReshape()*
 - MPBaseLinear()*
 - MPTanh()

*Use Initial components if they are at the top of the network and Base if they are below the top layers.
