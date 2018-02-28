----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'nn'
require 'optim'
require 'image'
require 'cutorch'
require 'cunn'
automation = require 'torchad_nn.datamodule'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-data', false, 'use full dataset (50,000 samples)')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-iterations', 1, 'maximum nb of iterations for SGD')
cmd:option('-threads', 1, 'nb of threads to use')
cmd:option('-usegpu', false, 'set true to train on gpu')
cmd:text()
opt = cmd:parse(arg)

sys.tic()

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- 10-classes
noutputs = #classes

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,100}

--size of conv filter.  Choose carefully so that filterpositions below are even
filtsize1 = 5
filtsize2 = 5

-- number of filter positions in each dim, assuming step 1. 
-- Outputs then squashed by factor of 2
filterPositions1 = width - filtsize1 + 1
convOutputs1 = filterPositions1/2
filterPositions2 = convOutputs1 - filtsize2 + 1
convOutputs2 = 1

-- output dim after 2 rounds of convolution
poolsize = 3

----------------------------------------------------------------------
-- a typical modern convolution network (conv+relu+pool)
model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize1, filtsize1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize2, filtsize2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : standard 2-layer neural network
-- todo - fix for new input and network sizes
model:add(nn.View(nstates[2]*convOutputs2*convOutputs2))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[2]*convOutputs2*convOutputs2, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3], noutputs))

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())


----------------------------------------------------------------------
-- get/create dataset
--
if opt.data then
   trsize = 32000
   tesize = 10000
else
   trsize = 4000
   tesize = 2000
end

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

-- Parallelizing
trainData.data, trainData.labels, trsize = automation.parallelize( trainData.data, trainData.labels, model, trsize, nil, nil, opt.batchSize) 

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   -- rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

setmetatable(trainData, 
    {__index = function(t, i) 
                    return {
                        t.data[i],
                        t.labels[i]
                    } 
                end})
				
setmetatable(testData, 
    {__index = function(t, i) 
                    return {
                        t.data[i],
                        t.labels[i]
                    } 
                end})

----------------------------------------------------------------------
-- and train!
--
criterion = nn.CrossEntropyCriterion() 

if opt.usegpu then
   print('<msg> setting model and dataset to GPU')
   model:cuda()
   trainData.data = trainData.data:cuda()
   testData.data = testData.data:cuda()
   criterion = criterion:cuda()
end

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = opt.learningRate
trainer.maxIteration = opt.iterations

-- record pre-processing time
t1 = sys.toc()

------------------------------------------------------------
-- train/test
--

-- train
sys.tic()
trainer:train(trainData)

-- record training time
t2 = sys.toc()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- perform test
for t = 1,testData:size() do
	local preds = model:forward(testData[t][1])
	confusion:add(preds, testData[t][2])
end

-- print confusion matrix
print(confusion)

local f = assert(io.open('SgdAuto--Size:' .. trainData:size() .. '--Batch:' .. opt.batchSize ..'.txt', "w"))

f:write('----------------------------------------\n')
f:write('----------------DataSet-----------------\n')
f:write('Train set Size:     ', trainData:size())
f:write('\n')
f:write('Test set Size:      ', testData:size())
f:write('\n')
f:write('batchSize:          ', opt.batchSize)
f:write('\n')
f:write('----------------Network-----------------\n')
f:write('Accuracy:           ', confusion.totalValid * 100)
f:write('\n')
f:write('Learning rate:      ', opt.learningRate)
f:write('\n')
f:write('----------------System------------------\n')
f:write('Threads:            ', opt.threads)
f:write('\n')
f:write('Pre-process Time:   ', t1)
f:write('\n')
f:write('Training Time:      ', t2)
f:write('\n')
f:write('----------------------------------------\n')

f:close()