# some hyper-parameters


from copy import deepcopy
# import required layer types
from mlp.layers import MLP, Linear, Sigmoid, Softmax
from mlp.layers import *
from mlp.optimisers import SGDOptimiser  # import the optimiser

# import the cost we want to use for optimisation
from mlp.costs import CECost, MSECost
from mlp.schedulers import LearningRateFixed
import numpy
import logging
from mlp.dataset import *


def base(train_dp, valid_dp, logger, learning_rate):
    # learning_rate = 0.01
    rng = numpy.random.RandomState([2016,02,26])

    max_epochs = 1000
    cost = CECost()

    stats = list()

    test_dp = deepcopy(valid_dp)
    train_dp.reset()
    valid_dp.reset()
    test_dp.reset()

    # NETWORK TOPOLOGY:
    model = MLP(cost=cost)
    model.add_layer(Relu(idim=125, odim=125, irange=1.6, rng=rng))
    model.add_layer(Softmax(idim=125, odim=19, rng=rng))


    # define the optimiser, here stochasitc gradient descent
    # with fixed learning rate and max_epochs
    lr_scheduler = LearningRateFixed(
        learning_rate=learning_rate, max_epochs=max_epochs)
    optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

    logger.info('Training started...')
    tr_stats_b, valid_stats_b = optimiser.train(model, train_dp, valid_dp)

    logger.info('Testing the model on test set:')

    tst_cost, tst_accuracy = optimiser.validate(model, test_dp)
    logger.info('ACL test set accuracy is %.2f %%, cost (%s) is %.3f' %
                (tst_accuracy*100., cost.get_name(), tst_cost))
