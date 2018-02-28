local datamodule = {}

local dataShuffle = false  --shuffle data when splitting
local syncPrototype = true --use sync model prototype

------------------------------------------------------------------
-- Name: 	parallelize
-- Inputs: 	data array, targets array, ANN model, data array size, mpi, mpinn, batchSize
-- Outputs: new data array, new targest array, optimized batch size, new data size
-- Summary: This function receives elements from the user and calls the necessary
--			functions to prepare data parallelization across the MPI nodes.
--
------------------------------------------------------------------

function datamodule.parallelize( data, targets, model, size, mpi_obj, mpinn_obj, batchSize )
	
	-- check required parameters were passed
	if (data == nil or targets == nil or model == nil or size == nil) then
		print ("-usage for parallelize(data, targets, model, size)")
		return -1
	end
	
	-- if no MPI object is passed create one
	if (mpi_obj == nil) then
		require 'torchmpi'
		mpi_obj = require('torchmpi')
		mpi_obj.start(true)  --true equals use GPU
	end
	
	-- if no MPI NN object is passed create one
	if (mpinn_obj == nil) then
		mpinn_obj = require('torchmpi.nn')
		mpinn_obj.synchronizeParameters(model)
	end
	
	-- split data and targets across all nodes
	local newdata, dataSize = datamodule.data_parallel(data, size, mpi_obj)
	local newtargets = datamodule.data_parallel(targets, size, mpi_obj)
	
	-- determine speed
	-- this option removed from current implementation
	-- local speed = datamodule.comm_speed(data, targets, model, mpi_obj, mpinn_obj)
	
	-- determine optimal batch size
	if batchSize ~= -1 then 
		datamodule.optimize_sync(speed, dataSize, model, mpinn_obj, mpi_obj, batchSize)
	end
	
	-- ensure all ranks are complete before returning
	mpi_obj.barrier()
	
	return newdata, newtargets, dataSize
end


------------------------------------------------------------------
-- Name: 	optimize_sync
-- Inputs: 	communication speed, data array size
-- Outputs: optimized batch size
-- Summary: This functions returns the optimized batch size based
--			on communication speed and data array size.
--
------------------------------------------------------------------

function datamodule.optimize_sync( speed, size, model, mpinn, mpi, batchSize )
	
	-- if batchSize is not given it will be optimized
	if (batchSize == nil) then
		if (size < 1000) then
			batchSize = 1
		elseif (size < 2500) then
			batchSize = 10
		elseif (size < 5000) then
			batchSize = 50
		else
			batchSize = 100
		end
	end
	
	--
	-- After comm test and batchSize determined override backward propogation function to syncGradients
	--
	function nn.Sequential:backward(input, gradOutput, scale)
		scale = scale or 1
		local currentGradOutput = gradOutput
		local currentModule = self.modules[#self.modules]
		for i=#self.modules-1,1,-1 do
			local previousModule = self.modules[i]
			currentGradOutput = self:rethrowErrors(currentModule, i+1, 'backward', previousModule.output, currentGradOutput, scale)
			currentModule.gradInput = currentGradOutput
			currentModule = previousModule
		end
		currentGradOutput = self:rethrowErrors(currentModule, 1, 'backward', input, currentGradOutput, scale)
		self.gradInput = currentGradOutput
		
		-- additional sync functionality added
		-- * 
		if (self.sync_counter == nil) then
			self.sync_counter = 1;
		end
		-- sync on batchSize and at the end of dataset
		if (self.sync_counter % batchSize == 0) or (self.sync_counter == size) then -- sync when batch is complete 
			if syncPrototype then
				datamodule.synchronizeModel(model, mpi)
			else
				mpinn.synchronizeGradients(model)
			end
		end
		self.sync_counter = self.sync_counter + 1
		-- *
		-- end of additional functionality
		
		return currentGradOutput
	end
	
	--This function needs to be overriden to allow for stochastic gradient training
	function nn.StochasticGradient:train(dataset)
	   local iteration = 1
	   local currentLearningRate = self.learningRate
	   local module = self.module
	   local criterion = self.criterion

	   local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
	   if not self.shuffleIndices then
		  for t = 1,dataset:size() do
			 shuffledIndices[t] = t
		  end
	   end

	   print("# StochasticGradient: training")

	   while true do
		  local currentError = 0
		  for t = 1,dataset:size() do
			 local example = dataset[shuffledIndices[t] ]
			 local input = example[1]
			 local target = example[2]

			 currentError = currentError + criterion:forward(module:forward(input), target)

			 module:updateGradInput(input, criterion:updateGradInput(module.output, target))
			 module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
			 
			-- additional sync functionality added
			-- * 
			if (self.sync_counter == nil) then
				self.sync_counter = 1
			end
			
			-- sync on batchSize and at the end of dataset
			if (self.sync_counter % batchSize == 0) or (self.sync_counter == size) then -- sync when batch is complete 
				if syncPrototype then
					datamodule.synchronizeModel(model, mpi)
				else
					mpinn.synchronizeGradients(model)
				end
			end
			self.sync_counter = self.sync_counter + 1
			-- *
			-- end of additional functionality

			 if self.hookExample then
				self.hookExample(self, example)
			 end
		  end

		  currentError = currentError / dataset:size()

		  if self.hookIteration then
			 self.hookIteration(self, iteration, currentError)
		  end

		  if self.verbose then
			 print("# current error = " .. currentError)
		  end
		  iteration = iteration + 1
		  currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
		  if self.maxIteration > 0 and iteration > self.maxIteration then
			 print("# StochasticGradient: you have reached the maximum number of iterations")
			 print("# training error = " .. currentError)
			 break
		  end
	   end
	end
	
	
	return batchSize
end

------------------------------------------------------------------
-- Name: 	synchronizeModel
-- Inputs: 	model, mpi handle
-- Outputs: none
-- Summary:	This function synchronizes model gradients and parameters
--			using allreduceTensor and collective operation average.
--
------------------------------------------------------------------

function datamodule.selectCollective(tensor, sync, collective, mpi)
   local ttype = torch.type(tensor):find('Cuda') and 'gpu' or 'cpu'
   if mpi.needInterNodeCollectives then
	  return assert(mpi.collectiveSelector[ttype].multinode[sync][collective],
	  'Could not find collective ' .. ttype .. ' multinode ' .. sync .. ' ' .. collective)
   else
	  return assert(mpi.collectiveSelector[ttype].singlenode[sync][collective],
	  'Could not find collective ' .. ttype .. ' singlenode ' .. sync .. ' ' .. collective)
   end
end

-- Synchronize model
function datamodule.synchronizeModel(net, mpi)
	if not net.parameters then return end
	local p, g = net:parameters()
	for i, w in ipairs(p) do
		local allreduceTensor = datamodule.selectCollective(w, 'sync', 'allreduceTensor', mpi)
		allreduceTensor(w)
		w:div(mpi.size())
	end
	for i, gw in ipairs(g) do
		local allreduceTensor = datamodule.selectCollective(gw, 'sync', 'allreduceTensor', mpi)
		allreduceTensor(gw)
		gw:div(mpi.size()) 
	end
end

------------------------------------------------------------------
-- Name: 	data_parallel
-- Inputs: 	array, array size
-- Outputs: new array, new size
-- Summary: This function splits the data array evenly across the MPI nodes.
--			Any remainder is given to the last node.
--
------------------------------------------------------------------

function datamodule.data_parallel( data, size, mpi )
	
	-- determine which rank will get which data
	-- copy data from dataset to newdataset from start to end
	local remainder = 0
	-- how many elements will be placed on each rank, remainder elements go to last rank
	local stripe = ( size - ( size % mpi.size() ) ) / mpi.size()
	size = size - remainder
	-- where will this rank's data start at
	local start  = ( mpi.rank() * stripe ) + 1
	-- where will this rank's data end at
	local finish = start + (stripe - 1) + remainder
	
	-- create local var for new dataset
	local newdata = {}
	
	newdata = data[{ {start,finish} } ]
	
	if dataShuffle then
		local index = 1
		for i = 1,size,mpi.size() do
			newdata[ { { index,index } } ] = data[ { { i + mpi.rank(), i + mpi.rank() } } ]
			index = index + 1
		end
	end
	
	print ("Rank: " .. mpi.rank() .. " Start Point: " .. start .. " End Point: " .. finish .. " Stripe: " .. stripe .. " Remainder: " .. remainder)
	
	-- return new dataset size
	local dataSize = stripe + remainder
	
	return newdata, dataSize
end
	

------------------------------------------------------------------
-- Name: 	comm_speed
-- Inputs: 	data array, target array, ANN model
-- Outputs: communication speed
-- Summary:	This function runs ten test of the network forward and backward
--			propogation with a sync of gradients across nodes. This is done
--			10 times and the average time returned.
--
------------------------------------------------------------------

function datamodule.comm_speed( data, targets, model, mpi, mpinn )

	-- ensure all ranks are ready before beginning comm check
	mpi.barrier() 
	
	local timer = torch.Timer()
	timer:stop()
	timer:resume()
	
	for i = 1,10 do 
		local output = model:forward(data[i])
		local df_do  = criterion:backward(output, targets[i])
		model:backward(data[i], df_do)
		mpinn.synchronizeGradients(model)
	end
	
	timer:stop()
	
	local speed = (timer:time().real) / 10
	
	print("Rank: " .. mpi.rank() .. " comm speed: " .. speed)
	
	return speed
end

return datamodule