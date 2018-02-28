

# Torch-Automatic-Distributed-Neural-Network
Torch Automatic Distributed Neural Network (TorchAD-NN) training library. Built on top of TorchMPI this module will automatically parallelize neural network training.

Main contribution is reducing the implementation complexity of data parallel neural network training by more than 90% and providing components, with near zero implementation complexity, to execute model parallel training on all or only select fully-connected neural layers.

See Thesis **insert link**

## Before Installation
Install [Torch 7.0](https://github.com/torch/torch7) and [TorchMPI](https://github.com/facebookresearch/TorchMPI)

**Note:** TorchMPI must be built using OpenMPI 2.0.1

## Installing from source
```bash
git clone https://github.com/ngrabaskas/Torch-Automatic-Distributed-Neural-Network.git
cd Torch-Automatic-Distributed-Neural-Network
luarocks make 
```

## To use Data Parallelism

Load TorchAD-NN library 
```lua
automation = require 'automatedparallelization.datamodule'
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
data, labels, size = automation.parallelize(data, labels, model, size, nil, nil, batchSize) 

----------------------------------------------------------------------
-- preprocess/normalize data
...

```

This parallelize() function will split dataset evenly across all nodes in the MPI handle. Synchronization will occur automatically as long as stochasticgradient:train() or model:backward() is being used in training.

To set synchronization manually use:
```lua 
-- insert example code
```

## To use Model Parallelism

Load TorchAD-NN library 
```lua
require 'automatedparallelization'
```

Use new neural layer component names.
```lua 
-- insert example code
```

## Training Concepts

__Performance__

* data should be 

... this will allocate 
