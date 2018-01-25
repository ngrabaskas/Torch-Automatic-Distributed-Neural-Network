# Torch-Automatic-Distributed-Neural-Network
Torch Automatic Distributed Neural Network (TAD-NN) training library. Built on top of TorchMPI this module will automatically parallelize neural network training.

## Installing from source
```bash
git clone https://github.com/ngrabaskas/Torch-Automatic-Distributed-Neural-Network
cd Torch-Automatic-Distributed-Neural-Network
luarocks make 
```

## To use

Simply convert your network model to CUDA by calling `:cuda()`:

```lua
local model = nn.Sequential()

```

... and similarly for your tensors:

## Training Concepts

__Performance__

* data should be 

... this will allocate 


