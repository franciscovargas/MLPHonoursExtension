import numpy
import logging
from mlp.dataset import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info('Initialising data providers...')

train_dp = ACLDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
valid_dp = ACLDataProvider(dset='valid', batch_size=1824, max_num_batches=1, randomize=False)


#Baseline experiment
from copy import deepcopy
from mlp.layers import MLP, Linear, Sigmoid, Softmax #import required layer types
from mlp.layers import *
from mlp.optimisers import SGDOptimiser #import the optimiser

from mlp.costs import CECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rng = numpy.random.RandomState([2015,10,10])

#some hyper-parameters
nhid = 100
learning_rate = 0.07
max_epochs = 30
cost = CECost()

stats = list()

test_dp = deepcopy(valid_dp)
train_dp.reset()
valid_dp.reset()
test_dp.reset()

#define the model
model = MLP(cost=cost)
#model.add_layer(ComplexLinear(idim=125, odim=125, irange=1.6, rng=rng))
#model.add_layer(Sigmoid(idim=2*125, odim=125, irange=1.6, rng=rng))
model.add_layer(Sigmoid(idim=125, odim=125, irange=1.6, rng=rng))
model.add_layer(Softmax(idim=125, odim=19, rng=rng))

# define the optimiser, here stochasitc gradient descent
# with fixed learning rate and max_epochs
lr_scheduler = LearningRateFixed(learning_rate=learning_rate, max_epochs=max_epochs)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

logger.info('Training started...')
tr_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)

logger.info('Testing the model on test set:')

tst_cost, tst_accuracy = optimiser.validate(model,test_dp )
logger.info('MNIST test set accuracy is %.2f %%, cost (%s) is %.3f'%(tst_accuracy*100., cost.get_name(), tst_cost))
