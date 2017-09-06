# Francisco Vargas
# Autoencoders and pretrainers here

import numpy
import matplotlib.pyplot as plt
import time
import logging
import numpy as np
from scipy.fftpack import fft

from mlp.layers import MLP
from mlp.layers import Sigmoid, Linear
from mlp.dataset import DataProvider
from mlp.schedulers import LearningRateScheduler
from mlp.costs import *
from copy import copy, deepcopy


logger = logging.getLogger(__name__)


class Optimiser(object):

    def train_epoch(self, model, train_iter):
        raise NotImplementedError()

    def train(self, model, train_iter, valid_iter=None):
        raise NotImplementedError()

    def validate(self, model, valid_iterator, l1_weight=0, l2_weight=0):

        acc_list, nll_list = [], []
        for x, t in valid_iterator:
            # print "what ???"
            y = model.fprop(x)
            nll_list.append(model.cost.cost(y, t))
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        acc = numpy.mean(acc_list)
        nll = numpy.mean(nll_list)

        prior_costs = Optimiser.compute_prior_costs(
            model, l1_weight, l2_weight)

        return nll + sum(prior_costs), acc


    def validate2(self, model, valid_iterator, l1_weight=0, l2_weight=0):

        acc_list, nll_list = [], []
        confidence, confusion = [] , []
        for x, t in valid_iterator:
            # print "what ???"
            y = model.fprop(x)
            nll_list.append(model.cost.cost(y, t))
            cls_acc = self.classification_accuracy(y, t)
            acc_list.append(numpy.mean(copy(cls_acc)))
            confidence.append(y)
            confusion.append(cls_acc)

        acc = numpy.mean(acc_list)
        nll = numpy.mean(nll_list)

        prior_costs = Optimiser.compute_prior_costs(
            model, l1_weight, l2_weight)

        return nll + sum(prior_costs), acc, confidence, confusion

    @staticmethod
    def classification_accuracy(y, t):
        """
        Returns classification accuracy given the estimate y and targets t
        :param y: matrix -- estimate produced by the model in fprop
        :param t: matrix -- target  1-of-K coded
        :return: vector of y.shape[0] size with binary values set to 0
                 if example was miscalssified or 1 otherwise
        """
        y_idx = numpy.argmax(y, axis=1)
        t_idx = numpy.argmax(t, axis=1)
        rval = numpy.equal(y_idx, t_idx)
        # print rval
        return rval

    @staticmethod
    def compute_prior_costs(model, l1_weight, l2_weight):
        """
        Computes the cost contributions coming from parameter-dependent only
        regularisation penalties
        """
        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )

        l1_cost, l2_cost = 0, 0
        for i in xrange(0, len(model.layers)):
            params = model.layers[i].get_params()
            for param in params:
                if l2_weight > 0:
                    l2_cost += 0.5 * l2_weight * numpy.sum(param**2)
                if l1_weight > 0:
                    l1_cost += l1_weight * numpy.sum(numpy.abs(param))

        return l1_cost, l2_cost


class SGDOptimiser(Optimiser):

    def __init__(self, lr_scheduler,
                 dp_scheduler=None,
                 l1_weight=0.0,
                 l2_weight=0.0):

        super(SGDOptimiser, self).__init__()

        assert isinstance(lr_scheduler, LearningRateScheduler), (
            "Expected lr_scheduler to be a subclass of 'mlp.schedulers.LearningRateScheduler'"
            " class but got %s " % type(lr_scheduler)
        )

        self.lr_scheduler = lr_scheduler
        self.dp_scheduler = dp_scheduler
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.f = 0

    def train_epoch(self, model, train_iterator, learning_rate):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )

        acc_list, nll_list = [], []
        self.f += 1
        for x, t in train_iterator:

            # get the prediction
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)

            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)

            # do backward pass through the model
            model.bprop(cost_grad, self.dp_scheduler)

            # update the model, here we iterate over layers
            # and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(0, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        # compute the prior penalties contribution (parameter dependent only)

        prior_costs = Optimiser.compute_prior_costs(
            model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    def train(self, model, train_iterator, valid_iterator=None, fft=False):

        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []

        # do the initial validation
        train_iterator.reset()
        tr_nll, tr_acc = self.validate(
            model, train_iterator, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: Training cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_nll, valid_acc = self.validate(
                model, valid_iterator, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: Validation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            valid_stats.append((valid_nll, valid_acc))

        if fft:
            from copy import deepcopy
            tmp = deepcopy(train_iterator)
            tmp.reset()
            model.layers[0].norm_weights(tmp)
        while not converged:
            train_iterator.reset()

            tstart = time.clock()
            tr_nll, tr_acc = self.train_epoch(model=model,
                                              train_iterator=train_iterator,
                                              learning_rate=self.lr_scheduler.get_rate())
            tstop = time.clock()
            if self.dp_scheduler is not None:
                self.dp_scheduler.get_next_rate()
            tr_stats.append((tr_nll, tr_acc))

            logger.info('Epoch %i: Training cost (%s) is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

            vstart = time.clock()
            if valid_iterator is not None:
                valid_iterator.reset()
                valid_nll, valid_acc = self.validate(model, valid_iterator,
                                                     self.l1_weight, self.l2_weight)
                logger.info('Epoch %i: Validation cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, valid_nll, valid_acc * 100.))
                self.lr_scheduler.get_next_rate(valid_acc)
                valid_stats.append((valid_nll, valid_acc))
            else:
                self.lr_scheduler.get_next_rate(None)
            vstop = time.clock()

            train_speed = train_iterator.num_examples_presented() / \
                (tstop - tstart)
            try:
                valid_speed = valid_iterator.num_examples_presented() / \
                    (vstop - vstart)
            except:
                valid_speed = float('inf')
            tot_time = vstop - tstart
            logger.info("Epoch %i: Took %.0f seconds. Training speed %.0f pps. "
                        "Validation speed %.0f pps."
                        % (self.lr_scheduler.epoch, tot_time, train_speed, valid_speed))

            # we stop training when learning rate, as returned by lr scheduler, is 0
            # this is implementation dependent and depending on lr schedule could happen,
            # for example, when max_epochs has been reached or if the progress between
            # two consecutive epochs is too small, etc.
            converged = (self.lr_scheduler.get_rate() == 0)

        return tr_stats, valid_stats

    @staticmethod
    def label_switch(train_iterator,
                     noise=lambda x: x):
        out = list()
        xp = list()
        tp = list()
        for x, t in train_iterator:
            xp.append(noise(x))
            tp.append(x)

        return zip(copy(xp), copy(tp))

    @staticmethod
    def fprop_label_switch(train_iterator,
                           model,
                           noise=lambda x: x):
        out = list()
        tp = list()
        xp = list()
        for x, t in train_iterator:
            xp.append(noise(x))
            tp.append(model.fprop(x))
        return zip(xp, tp)

    def pretrain_epoch(self, model, train_iterator,
                       learning_rate, fprop_list, to_layer=0,
                       final=False, noise_up_layer=-1, noise=False):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )

        acc_list, nll_list = [], []

        if fprop_list is not None:
            train_iterator = fprop_list
        for x, t in train_iterator:
            if(fprop_list is not None):
                t2 = t
            elif not final:
                t2 = x
            else:
                t2 = t

            if noise:
                x *= self.noise_stack[0]

            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x, noise_up_layer=noise_up_layer,
                                noise_list=self.noise_stack)

            # compute the cost and grad of the cost w.r.t y

            cost = model.cost.cost(y, t2)

            cost_grad = model.cost.grad(y, t2)

            # do backward pass through the model
            model.bprop(cost_grad, self.dp_scheduler)

            # update the model, here we iterate over layers
            # and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]
            if not final:
                assert (len(model.layers) - to_layer == 2)
            else:
                assert (len(model.layers) - to_layer == 1)
            for i in xrange(to_layer, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t2)))

        # compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(
            model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    def pretrain(self, model, train_iterator,
                 valid_iterator=None, noise=False):
        """
        Returns the layers. Since I was a bit scared of making it return the model
        and then ran out of time when it came to cleaning up this code....
        the code base was super cool but it was time consuming workinground things
        here and there
        """

        # Whilst the slides say not to noise the learned represantations when
        # Carrying out a denoising autoencoder in eached learned representation
        # makes sense when inductively carrying out the definition of a single
        # unit autoencoder. Nonetheless I did it in the way the slides point out.
        # yet my version can still be run by passing wrong=True to fprop
        self.noise_stack = [model.rng.binomial(
            1, 0.25, (train_iterator.batch_size, f.odim)) for f in model.layers]
        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []

        cost = MSECost()
        model_out = MLP(cost=cost)
        init_layer = Sigmoid(
            idim=model.layers[0].idim, odim=model.layers[0].odim, rng=model.rng)

        model_out.add_layer(init_layer)
        output_layer = Linear(
            idim=init_layer.odim, odim=125, rng=init_layer.rng)

        model_out.add_layer(output_layer)

        # do the initial validation
        train_iterator.reset()
        train_iterator_tmp = self.label_switch(train_iterator)
        tr_nll, tr_acc = self.validate(
            model_out, train_iterator_tmp, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: PreTraining cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_iterator_tmp = self.label_switch(valid_iterator)
            valid_nll, valid_acc = self.validate(
                model_out, valid_iterator_tmp, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: PreValidation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            valid_stats.append((valid_nll, valid_acc))

        layers = model.layers
        layers_out = list()
        fprop_list = None
        print len(layers)

        final = False

        noise_layer = -1
        if noise:
            noise_layer = 0
        for to_layer in range(len(layers)):
            #  This is very ugly yes but I invested my time in conv
            if(to_layer > 0 and len(layers) > 2 and to_layer < len(layers) - 1):

                train_iterator.reset()
                model_out.remove_top_layer()
                fprop_list = self.fprop_label_switch(
                    (train_iterator), model_out)

                if noise:
                    noise_layer = to_layer
                tmp_layer = Sigmoid(idim=model_out.layers[len(model_out.layers) - 1].odim,
                                    odim=layers[to_layer].odim,
                                    rng=init_layer.rng)
                model_out.add_layer(tmp_layer)
                output_layer = Linear(idim=tmp_layer.odim,
                                      odim=tmp_layer.idim,
                                      rng=init_layer.rng)
                model_out.add_layer(output_layer)

            elif to_layer == len(layers) - 1:
                final = True
                train_iterator.reset()

                model_out.remove_top_layer()
                fprop_list = None

                output_layer = layers[-1]
                model_out.add_layer(output_layer)
                model_out.cost = CECost()
                noise_layer = -1

            while not converged:
                train_iterator.reset()

                tstart = time.clock()
                tr_nll, tr_acc = self.pretrain_epoch(model=model_out,
                                                     train_iterator=(
                                                         train_iterator),
                                                     learning_rate=self.lr_scheduler.get_rate(),
                                                     to_layer=to_layer,
                                                     fprop_list=fprop_list,
                                                     final=final,
                                                     noise_up_layer=noise_layer)
                tstop = time.clock()
                tr_stats.append((tr_nll, tr_acc))

                logger.info('Epoch %i: PreTraining cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

                vstart = time.clock()
                if valid_iterator is not None:
                    valid_iterator.reset()
                    if fprop_list is not None:
                        valid_iterator_tmp = fprop_list
                    elif not final:
                        valid_iterator_tmp = self.label_switch(valid_iterator)
                    else:
                        valid_iterator_tmp = valid_iterator
                    valid_nll, valid_acc = self.validate(model_out, valid_iterator_tmp,
                                                         self.l1_weight, self.l2_weight)
                    logger.info('Epoch %i: PreValidation cost (%s) is %.3f. Accuracy is %.2f%%'
                                % (self.lr_scheduler.epoch + 1, cost_name, valid_nll, valid_acc * 100.))
                    self.lr_scheduler.get_next_rate(valid_acc)
                    valid_stats.append((valid_nll, valid_acc))
                else:
                    self.lr_scheduler.get_next_rate(None)
                vstop = time.clock()

                train_speed = train_iterator.num_examples_presented() / \
                    (tstop -
                     tstart)
                valid_speed = valid_iterator.num_examples_presented() / \
                    (vstop -
                     vstart)
                tot_time = vstop - tstart
                # pps = presentations per second
                logger.info("Epoch %i: Took %.0f seconds. PreTraining speed %.0f pps. "
                            "Validation speed %.0f pps."
                            % (self.lr_scheduler.epoch, tot_time, train_speed, valid_speed))

                converged = (self.lr_scheduler.get_rate() == 0)
            # reseting epochs to zero could have just done lr_shed.epoch =0
            # but I foucsed most my time on cleaning up the conv code
            self.lr_scheduler.epoch = 0
            if self.dp_scheduler is not None:
                self.dp_scheduler.epoch = 0
            converged = False
            layers_out.append(model_out.layers[to_layer])

        return layers_out, tr_stats, valid_stats

    def pretrain_discriminative(self, model, train_iterator, valid_iterator=None):

        converged = False
        cost_name = model.cost.get_name()
        tr_stats, valid_stats = [], []
        layer_num = len(model.layers)
        cost = CECost()
        model_out = MLP(cost=cost)
        init_layer = model.layers[0]

        layers = model.layers
        model_out.add_layer(init_layer)

        bleugh = layers_dict[layers[-1].get_name()](idim=init_layer.odim,
                                                    odim=layers[-1].odim, rng=model.rng, irange=layers[-1].irange)

        model_out.add_layer((layers_dict[layers[-1].get_name()](idim=init_layer.odim,
                                                                odim=layers[-1].odim, rng=model.rng, irange=layers[-1].irange)))

        # do the initial validation
        train_iterator.reset()

        tr_nll, tr_acc = self.validate(
            model_out, train_iterator, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: Training cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, cost_name, tr_nll, tr_acc * 100.))
        tr_stats.append((tr_nll, tr_acc))

        if valid_iterator is not None:
            valid_iterator.reset()
            valid_nll, valid_acc = self.validate(
                model, valid_iterator, self.l1_weight, self.l2_weight)
            logger.info('Epoch %i: Validation cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch, cost_name, valid_nll, valid_acc * 100.))
            valid_stats.append((valid_nll, valid_acc))

        for to_layer in range(len(layers)):
            if(to_layer > 0 and len(layers) > 2 and to_layer < len(layers) - 1):
                model_out.remove_top_layer()
                model_out.layers[
                    len(model_out.layers) - 1].odim = layers[to_layer].idim
                tmp_layer = copy(layers[to_layer])
                model_out.add_layer(tmp_layer)
                # This is here to allow the final layer having a different dim
                # to the hidden layers output since the weight matrix needs
                # to be reshaped and reinstantiated. Thus I had to modify code
                # in layers.py to cater for the global variables that allowed me
                # to do so. I believe this may have been overlooked
                model_out.add_layer(layers_dict[layers[-1].get_name()](idim=tmp_layer.odim,
                                                                       odim=layers[-1].odim, rng=model.rng, irange=layers[-1].irange))
            while not converged:
                train_iterator.reset()

                tstart = time.clock()

                tr_nll, tr_acc = self.pretrain_discriminative_epoch(model=model_out,
                                                                    train_iterator=train_iterator,
                                                                    learning_rate=self.lr_scheduler.get_rate(),
                                                                    to_layer=to_layer)
                tstop = time.clock()
                tr_stats.append((tr_nll, tr_acc))

                logger.info('Epoch %i: PreTraining cost (%s) is %.3f. Accuracy is %.2f%%'
                            % (self.lr_scheduler.epoch + 1, cost_name, tr_nll, tr_acc * 100.))

                vstart = time.clock()
                if valid_iterator is not None:
                    valid_iterator.reset()
                    valid_nll, valid_acc = self.validate(model, valid_iterator,
                                                         self.l1_weight, self.l2_weight)
                    logger.info('Epoch %i: PreValidation cost (%s) is %.3f. Accuracy is %.2f%%'
                                % (self.lr_scheduler.epoch + 1, cost_name, valid_nll, valid_acc * 100.))
                    self.lr_scheduler.get_next_rate(valid_acc)
                    valid_stats.append((valid_nll, valid_acc))
                else:
                    self.lr_scheduler.get_next_rate(None)
                vstop = time.clock()

                train_speed = train_iterator.num_examples_presented() / \
                    (tstop -
                     tstart)
                valid_speed = valid_iterator.num_examples_presented() / \
                    (vstop -
                     vstart)
                tot_time = vstop - tstart
                # pps = presentations per second
                logger.info("Epoch %i: Took %.0f seconds. PreTraining speed %.0f pps. "
                            "Validation speed %.0f pps."
                            % (self.lr_scheduler.epoch, tot_time, train_speed, valid_speed))

                # we stop training when learning rate, as returned by lr scheduler, is 0
                # this is implementation dependent and depending on lr schedule could happen,
                # for example, when max_epochs has been reached or if the progress between
                # two consecutive epochs is too small, etc.
                converged = (self.lr_scheduler.get_rate() == 0)
            self.lr_scheduler = copy(self.cache_l)
            self.dp_scheduler = copy(self.cache_d)
            converged = False

        return model_out, tr_stats, valid_stats

    def pretrain_discriminative_epoch(self, model, train_iterator, learning_rate, to_layer):

        assert isinstance(model, MLP), (
            "Expected model to be a subclass of 'mlp.layers.MLP'"
            " class but got %s " % type(model)
        )
        assert isinstance(train_iterator, DataProvider), (
            "Expected iterator to be a subclass of 'mlp.dataset.DataProvider'"
            " class but got %s " % type(train_iterator)
        )

        acc_list, nll_list = [], []

        for x, t in train_iterator:

            # get the prediction
            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)

            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)

            # do backward pass through the model
            model.bprop(cost_grad, self.dp_scheduler, to_layer=to_layer)

            # update the model, here we iterate over layers
            # and then over each parameter in the layer
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(to_layer, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        # compute the prior penalties contribution (parameter dependent only)
        prior_costs = Optimiser.compute_prior_costs(
            model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    @staticmethod
    def fft_label_switch(train_iterator,
                         noise=lambda x: x):
        out = list()
        tp = list()
        xp = list()
        for x, t in train_iterator:
            xp.append(noise(x))
            tp.append(np.abs(fft(x)))
        return zip(xp, tp)

    def spretrain_epoch(self, model, train_iterator,
                        learning_rate, fprop_list, to_layer=0):

        acc_list, nll_list = [], []

        if fprop_list is not None:
            train_iterator = fprop_list
        for x, t in train_iterator:

            if self.dp_scheduler is not None:
                y = model.fprop_dropout(x, self.dp_scheduler)
            else:
                y = model.fprop(x)

            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)
            model.bprop(cost_grad, self.dp_scheduler)
            effective_learning_rate = learning_rate / x.shape[0]

            for i in xrange(to_layer, len(model.layers)):
                params = model.layers[i].get_params()
                grads = model.layers[i].pgrads(inputs=model.activations[i],
                                               deltas=model.deltas[i + 1],
                                               l1_weight=self.l1_weight,
                                               l2_weight=self.l2_weight)
                uparams = []
                for param, grad in zip(params, grads):
                    param = param - effective_learning_rate * grad
                    uparams.append(param)
                model.layers[i].set_params(uparams)

            nll_list.append(cost)
            acc_list.append(numpy.mean(self.classification_accuracy(y, t)))

        prior_costs = Optimiser.compute_prior_costs(
            model, self.l1_weight, self.l2_weight)
        training_cost = numpy.mean(nll_list) + sum(prior_costs)

        return training_cost, numpy.mean(acc_list)

    def spretrain(self, model, train_iterator,
                  valid_iterator=None, noise=False):

        self.noise_stack = [model.rng.binomial(1, 0.25,
                                               (train_iterator.batch_size,
                                                f.odim)) for f in model.layers]
        converged = False
        tr_stats, valid_stats = [], []

        cost = MSECost()
        model_out = MLP(cost=cost)

        init_layer = Linear(idim=model.layers[0].idim,
                            odim=model.layers[0].idim*2,
                            rng=model.rng)
        model_out.add_layer(init_layer)
        nl_layer = Sigmoid(idim=model.layers[0].idim*2,
                           odim=model.layers[0].idim,
                           rng=model.rng)
        model_out.add_layer(nl_layer)
        output_layer = Linear(idim=model.layers[0].idim,
                              odim=model.layers[0].idim,
                              rng=model.rng)
        model_out.add_layer(output_layer)

        # do the initial validation
        train_iterator.reset()
        train_iterator_tmp = self.label_switch(train_iterator)
        tr_nll, tr_acc = self.validate(
            model_out, train_iterator_tmp, self.l1_weight, self.l2_weight)
        logger.info('Epoch %i: SpecPreTraining cost (%s) for initial model is %.3f. Accuracy is %.2f%%'
                    % (self.lr_scheduler.epoch, model_out.cost.get_name(), tr_nll, tr_acc * 100.))
        tr_stats.append((tr_nll, tr_acc))

        layers = model.layers
        layers_out = list()


        train_iterator.reset()
        fprop_list = self.fft_label_switch(deepcopy(train_iterator))
        # print fprop_list
        while not converged:
            train_iterator.reset()

            tstart = time.clock()
            tr_nll, tr_acc = self.spretrain_epoch(model=model_out,
                                                  train_iterator=(train_iterator),
                                                  learning_rate=self.lr_scheduler.get_rate(),
                                                  to_layer=0,
                                                  fprop_list=fprop_list)
            tstop = time.clock()
            tr_stats.append((tr_nll, tr_acc))

            logger.info('Epoch %i: PreTraining cost (%s) is %.3f. Accuracy is %.2f%%'
                        % (self.lr_scheduler.epoch + 1, model_out.cost.get_name(), tr_nll, tr_acc * 100.))

            self.lr_scheduler.get_next_rate(None)
            vstop = time.clock()

            train_speed = train_iterator.num_examples_presented() / \
                (tstop - tstart)
            tot_time = vstop - tstart

            converged = (self.lr_scheduler.get_rate() == 0)
        # reseting epochs to zero could have just done lr_shed.epoch =0
        # but I foucsed most my time on cleaning up the conv code

        return model_out, tr_stats, valid_stats
