local nodemodule = {}

local printDims   = false  -- For debugging information
local syncTanh    = false  -- Experimental tanh all gather
local syncReshape = false  -- Experimental reshape all gather

------------------------------------------------------------------
-- Name: 	Narrow Input
-- Inputs: 	Input array
-- Outputs: Narrowed input array
-- Summary: This function accepts the normal input array and returns the
--			the reduced array that is specific to each node.
--
------------------------------------------------------------------

local function narrowInput(input)
   local dim = input:nDimension()
   assert(input:size(dim) % mpi.size() == 0)
   local size = input:size(dim) / mpi.size()
   return input:narrow(dim, mpi.rank() * size + 1, size)
end

------------------------------------------------------------------
-- Name: 	Initial Linear Layer
-- Inputs: 	No additional input is required to use this layer
-- Outputs: Standard output from nn.Linear is used
-- Summary: These functions override the original and allow the input layer to
--			be split over multiple nodes within the network. Output is then 
--			Synchronized using allreduceTensor(). For the initial linear layer
--			accGradParameters() input must also be narrowed.
--
------------------------------------------------------------------

local MPInitialLinear, parent = torch.class('nn.MPInitialLinear', 'nn.Linear')
function MPInitialLinear.__init(self, i, o)
   assert(i % mpi.size() == 0, ('i=%d not divisible by %d'):format(i, mpi.size()))
   nn.Linear.__init(self, i / mpi.size(), o)
end

function MPInitialLinear.updateOutput(self, input)
	local input = narrowInput(input)
	if printDims then
		print('#INITIAL LINEAR UPDATE OUTPUT#')
		print('*****input*****')
		print(input:size())
	end
	self.output = nn.Linear.updateOutput(self, input)
      if printDims then
		print('*****output*****')
		print(self.output:size())
	end
	mpi.allreduceTensor(self.output)
	return self.output
end

function MPInitialLinear.updateGradInput(self, input, gradOutput)
	local input = narrowInput(input)
	if printDims then
		print('#INITIAL LINEAR UPDATE GRAD INPUT#')
		print('*****input*****')
		print(input:size())
		print('*****grad output*****')
		print(gradOutput:size())
	end
	self.gradInput = nn.Linear.updateGradInput(self, input, gradOutput)
	mpi.allreduceTensor(self.gradInput)
	return self.gradInput
end

function MPInitialLinear.accGradParameters(self, input, gradOutput, scale)
   local input = narrowInput(input)
   nn.Linear.accGradParameters(self, input, gradOutput, scale)
end

------------------------------------------------------------------
-- Name: 	Base Linear Layer
-- Inputs: 	No additional input is required to use this layer
-- Outputs: Standard output from nn.Linear is used
-- Summary: These functions override the original and allow the input layer to
--			be split over multiple nodes within the network. Output is then 
--			Synchronized using allreduceTensor(). For base linear layer, 
--			accGradParameters() input is not narrowed.
--
------------------------------------------------------------------

local MPBaseLinear, parent = torch.class('nn.MPBaseLinear', 'nn.Linear')
function MPBaseLinear.__init(self, i, o)
   assert(i % mpi.size() == 0, ('i=%d not divisible by %d'):format(i, mpi.size()))
   dimension_used = i / mpi.size()
   nn.Linear.__init(self, dimension_used, o)
end

function MPBaseLinear.updateOutput(self, input)
   local input = narrowInput(input)
   self.output = nn.Linear.updateOutput(self, input)
   if printDims then
		print('#BASE LINEAR UPDATE OUTPUT#')
		print('*****input*****')
		print(input:size())
		print('*****output*****')
		print(self.output:size())
	end
	mpi.allreduceTensor(self.output)
	return self.output
end

function MPBaseLinear.updateGradInput(self, input, gradOutput)
	local input = narrowInput(input)
	if printDims then
		print('#BASE LINEAR UPDATE GRAD INPUT#')
		print('*****input*****')
		print(input:size())
		print('*****grad output*****')
		print(gradOutput:size())
	end
	self.gradInput = nn.Linear.updateGradInput(self, input, gradOutput)
	mpi.allreduceTensor(self.gradInput)
	return self.gradInput
end

------------------------------------------------------------------
-- Name: 	Tanh Layer
-- Inputs: 	No additional input is required to use this layer
-- Outputs: Standard output from nn.Tanh is used
-- Summary: These functions narrow the self.output from updateGradInput()
--			during backward propagation. Output from updateGradInput() is
--			expanded for the above linear layer during backward propagation.
--			An experiment allgather during this expansion can be turned on.
--
------------------------------------------------------------------

local MPTanh, parent = torch.class('nn.MPTanh', 'nn.Tanh')
function MPTanh:updateOutput(input)
	self.output = nn.Tanh.updateOutput(self, input)
	if printDims then
		print('#TANH UPDATE OUTPUT#')
		print('*****input*****')
		print(input:size())
		print('*****output*****')
		print(self.output:size())
	end
	return self.output
end

function MPTanh:updateGradInput(input, gradOutput)
	self.output = narrowInput(self.output)
	if printDims then
		print('#TANH UPDATE GRAD INPUT#')
		print('*****input*****')
		print(input:size())
		print('*****output*****')
		print(self.output:size())
	end
	input.THNN.Tanh_updateGradInput(
	  input:cdata(),
	  gradOutput:cdata(),
	  self.gradInput:cdata(),
	  self.output:cdata()
	)
	-- need to get gradInput from each node and concatenate ######
	tempGradInput = self.gradInput
	for i=2,mpi.size() do
		tempGradInput = torch.cat(tempGradInput, self.gradInput)
	end
	if mpi.size() >  1 and syncTanh then
		mpi.allgatherTensor(self.gradInput, tempGradInput)
	end
	self.gradInput = tempGradInput
	if printDims then
		print('*****grad input*****')
		print(self.gradInput:size())
	end
	return self.gradInput
end

------------------------------------------------------------------
-- Name: 	Initial Reshape Layer
-- Inputs: 	No additional input is required to use this layer
-- Outputs: Standard output from nn.Reshape is used
-- Summary: The below functions narrows the input when Reshape.updateGradInput is
-- 			used at the top of the network. This corresponds to the 
--			narrowed out from the below linear layer's updateGradInput().
--
------------------------------------------------------------------

local MPInitialReshape, parent = torch.class('nn.MPInitialReshape', 'nn.Reshape')

function MPInitialReshape:updateOutput(input)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end

   if (self.batchMode == false) or (
         (self.batchMode == nil) and
         (input:nElement() == self.nelement and input:size(1) ~= 1)
      ) then
      self.output:view(input, self.size)
   else
      self.batchsize[1] = input:size(1)
      self.output:view(input, self.batchsize)
   end
   return self.output
end

function MPInitialReshape:updateGradInput(input, gradOutput)
	local input = narrowInput(input)
	if printDims then
		print('#INITIAL RESHAPE UPDATE GRAD INPUT#')
		print('*****input*****')
		print(input:size())
		print('*****output*****')
		print(gradOutput:size())
	end
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end

   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end

------------------------------------------------------------------
-- Name: 	Base Reshape Layer
-- Inputs: 	No additional input is required to use this layer
-- Outputs: Standard output from nn.Reshape is used
-- Summary: Output from updateGradInput() is expanded for the above linear 
--			layer during backward propagation. An experiment allgather 
--			during this expansion can be turned on.
--
------------------------------------------------------------------

local MPBaseReshape, parent = torch.class('nn.MPBaseReshape', 'nn.Reshape')

function MPBaseReshape:updateOutput(input)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end

   if (self.batchMode == false) or (
         (self.batchMode == nil) and
         (input:nElement() == self.nelement and input:size(1) ~= 1)
      ) then
      self.output:view(input, self.size)
   else
      self.batchsize[1] = input:size(1)
      self.output:view(input, self.batchsize)
   end
   return self.output
end

function MPBaseReshape:updateGradInput(input, gradOutput)
	tempGrad = gradOutput
	for i=2,mpi.size() do
		tempGrad = torch.cat(tempGrad, gradOutput)
	end
	if mpi.size() >  1 and syncReshape then
		mpi.allgatherTensor(gradOutput, tempGrad)
	end
	gradOutput = tempGrad
	if printDims then
		print('#BASE RESHAPE UPDATE GRAD INPUT#')
		print('*****input*****')
		print(input:size())
		print('*****output*****')
		print(gradOutput:size())
	end
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end

   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end

return nodemodule


   
