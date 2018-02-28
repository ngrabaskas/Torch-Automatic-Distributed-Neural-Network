package = "torchad_nn"
version = "scm-1"

source = {
   url = "https://github.com/ngrabaskas/Torch-Automatic-Distributed-Neural-Network.git",
   tag = "master"
}

description = {
   summary = "Torch Automatic Distributed Neural Network (TorchAD-NN) training library. Built on top of TorchMPI this module will automatically parallelize neural network training.",
   detailed = [[
   	    Automated Parallelization for Torch and TorchMPI
   ]],
   homepage = "https://github.com/ngrabaskas/Torch-Automatic-Distributed-Neural-Network"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
