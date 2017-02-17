import numpy
import logging
from mlp.dataset import *
from mlp.layers import *  #import required layer types
import numpy
import logging
from mlp.layers import MLP, Linear #import required layer types
from mlp.optimisers import SGDOptimiser #import the optimiser
from mlp.dataset import * #import data provider
from mlp.costs import  *
# import MSECost, CECost #import the cost we want to use for optimisation
from mlp.schedulers import LearningRateFixed
import pandas
from copy import deepcopy
import cPickle as p
import time
from mlp.convlin import *

if __name__ == "__main__":
    print "enter"
    seeds = list()
    start = time.time()
    train_dp = MACLDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True,name='RLAx')
    valid_dp = MACLDataProvider(dset='test', batch_size=1140, max_num_batches=1, randomize=False,name='RLAx')
    test_dp = MACLDataProvider(dset='valid', batch_size=1140, max_num_batches=1, randomize=False,name='RLAx')
    for dt in pandas.date_range("2015-01-10","2015-10-10"):

        print "date: " +  str(dt)
        train_dp.reset()
        test_dp.reset()
        valid_dp.reset()



        rng = numpy.random.RandomState([dt.year,dt.month,dt.day])

        # define the model structure, here just one linear layer
        # and mean square error cost
        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(ComplexAbs(idim=125, odim=125))
        model.add_layer(ConvRelu_Opt( 1, 1, rng=rng, stride=(1,1)  ))
        model.add_layer(Sigmoid(idim=122, odim=122, rng=rng))
        model.add_layer(Softmax(idim=122, odim=19, rng=rng))
        #one can stack more layers here


        # print map(lambda x: (x.idim, x.odim), model.layers)
        lr_scheduler = LearningRateFixed(learning_rate=0.01, max_epochs=500)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

        tr_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)
        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)
        seeds.append((tr_stats, valid_stats, (tst_cost, tst_accuracy)))

    end = time.time()
    print"scipy.correlate time: " + str(end - start)
    with open('seeds_conv_fft_final.pkl','wb') as f:
        p.dump(seeds, f)
