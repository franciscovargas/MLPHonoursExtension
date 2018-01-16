# Francisco Vargas

from scipy import signal

from sklearn.preprocessing import normalize

import numpy
import logging
from mlp.costs import Cost
import matplotlib.pyplot as plt
from scipy.linalg import dft


logger = logging.getLogger(__name__)


def max_and_argmax(x, axes=None, keepdims_max=False, keepdims_argmax=False):
    """
    Return both max and argmax for the given multi-dimensional array, possibly
    preserve the original shapes
    :param x: input tensor
    :param axes: tuple of ints denoting axes across which
                 one should perform reduction
    :param keepdims_max: boolean, if true, shape of x is preserved in result
    :param keepdims_argmax:, boolean, if true, shape of x is preserved in result
    :return: max (number) and argmax (indices) of max element along certain axes
             in multi-dimensional tensor
    """
    if axes is None:
        rval_argmax = numpy.argmax(x)
        if keepdims_argmax:
            rval_argmax = numpy.unravel_index(rval_argmax, x.shape)
    else:
        if isinstance(axes, int):
            axes = (axes,)
        axes = tuple(axes)
        keep_axes = numpy.array([i for i in range(x.ndim) if i not in axes])
        transposed_x = numpy.transpose(x, numpy.concatenate((keep_axes, axes)))
        reshaped_x = transposed_x.reshape(
            transposed_x.shape[:len(keep_axes)] + (-1,))
        rval_argmax = numpy.asarray(
            numpy.argmax(reshaped_x, axis=-1), dtype=numpy.int64)

        # rval_max_arg keeps the arg index referencing to the axis along which reduction was performed (axis=-1)
        # when keepdims_argmax is True we need to map it back to the original shape of tensor x
        # print 'rval maxaarg', rval_argmax.ndim, rval_argmax.shape,
        # rval_argmax
        if keepdims_argmax:
            dim = tuple([x.shape[a] for a in axes])
            rval_argmax = numpy.array([idx + numpy.unravel_index(val, dim)
                                       for idx, val in numpy.ndenumerate(rval_argmax)])
            # convert to numpy indexing convention (row indices first, then
            # columns)
            rval_argmax = zip(*rval_argmax)

    if keepdims_max is False and keepdims_argmax is True:
        # this could potentially save O(N) steps by not traversing array once more
        # to get max value, haven't benchmark it though
        rval_max = x[rval_argmax]
    else:
        rval_max = numpy.asarray(
            numpy.amax(x, axis=axes, keepdims=keepdims_max))

    return rval_max, rval_argmax


class MLP(object):

    """
    This is a container for an arbitrary sequence of other transforms
    On top of this, the class also keeps the state of the model, i.e.
    the result of forward (activations) and backward (deltas) passes
    through the model (for a mini-batch), which is required to compute
    the gradients for the parameters
    """

    def __init__(self, cost, rng=None):

        assert isinstance(cost, Cost), (
            "Cost needs to be of type mlp.costs.Cost, got %s" % type(cost)
        )

        self.layers = []  # the actual list of network layers
        # keeps forward-pass activations (h from equations)
        self.activations = []
        # for a given minibatch (or features at 0th index)
        # keeps back-propagated error signals (deltas from equations)
        self.deltas = []
        # for a given minibatch and each layer
        self.cost = cost

        if rng is None:
            self.rng = numpy.random.RandomState([2015, 11, 11])
        else:
            self.rng = rng
            # print "rng inherited"

    def fprop(self, x, noise_up_layer=-1, noise_list=None, wrong=False):
        """

        :param inputs: mini-batch of data-points x
        :return: y (top layer activation) which is an estimate of y given x
        """

        if len(self.activations) != len(self.layers) + 1:
            self.activations = [None]*(len(self.layers) + 1)

        self.activations[0] = x
        # print "layers: >>>>" + str(len(self.activations))
        # print "layer len: ", len(self.layers)
        # if noise_list is not None:
        #     print "noise stack len: ", len(noise_list)
        for i in xrange(0, len(self.layers)):
            # print "layer : " + str(i)
            # print "odim : " + str(self.layers[i].odim)
            # print "idim : " + str(self.layers[i].idim)
            # print "activation : " + str(self.activations[i].shape)
            self.activations[i+1] = self.layers[i].fprop(self.activations[i])
            if noise_up_layer != -1 and i < len(self.layers) - 1:
                # print "Layer: ", i
                # nomn pythonic hack prepare your ...
                # this is moribidly obese
                if wrong:
                    self.activations[i+1] *= noise_list[i]
                # pass

        return self.activations[-1]

    def fprop_dropout(self, x, dp_scheduler):
        """
        :param inputs: mini-batch of data-points x
        :param dp_scheduler: dropout scheduler
        :return: y (top layer activation) which is an estimate of y given x
        """

        if len(self.activations) != len(self.layers) + 1:
            self.activations = [None]*(len(self.layers) + 1)

        p_inp, p_hid = dp_scheduler.get_rate()

        d_inp = 1
        p_inp_scaler, p_hid_scaler = 1.0/p_inp, 1.0/p_hid
        if p_inp < 1:
            d_inp = self.rng.binomial(1, p_inp, size=x.shape)

        # it's OK to scale the inputs by p_inp_scaler here
        self.activations[0] = p_inp_scaler*d_inp*x
        self.activations[1] = self.layers[0].fprop(self.activations[0])
        for i in xrange(1, len(self.layers)):
            d_hid = 1
            if p_hid < 1:
                d_hid = self.rng.binomial(
                    1, p_hid, size=self.activations[i].shape)
            # but not the hidden activations, since the non-linearity grad
            # *may* explicitly depend on them
            self.activations[i] *= d_hid
            self.activations[
                i+1] = self.layers[i].fprop(p_hid_scaler*self.activations[i])

        return self.activations[-1]

    def bprop(self, cost_grad, dp_scheduler=None):
        """
        :param cost_grad: matrix -- grad of the cost w.r.t y
        :return: None, the deltas are kept in the model
        """

        # allocate the list of deltas for each layer
        # note, we do not use all of those fields but
        # want to keep it aligned 1:1 with activations,
        # which will simplify indexing later on when
        # computing grads w.r.t parameters
        if len(self.deltas) != len(self.activations):
            self.deltas = [None]*len(self.activations)

        # treat the top layer in special way, as it deals with the
        # cost, which may lead to some simplifications
        top_layer_idx = len(self.layers)
        self.deltas[top_layer_idx], ograds = self.layers[top_layer_idx - 1].\
            bprop_cost(self.activations[top_layer_idx], cost_grad, self.cost)


        p_hid_scaler = 1.0
        if dp_scheduler is not None:
            p_inp, p_hid = dp_scheduler.get_rate()
            p_hid_scaler /= p_hid

        # then back-prop through remaining layers
        for i in xrange(top_layer_idx - 1, 0, -1):
            # print ograds.shape, "!hhhhhhhhhhh",( ograds*p_hid_scaler).shape,  self.layers[i - 1], self.activations[i]
            self.deltas[i], ograds = self.layers[i - 1].\
                bprop(self.activations[i], ograds*p_hid_scaler)

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_layers(self, layers):
        self.layers = layers

    def get_name(self):
        return 'mlp'

    def remove_top_layer(self):
        self.layers.pop(-1)


class Layer(object):

    """
    Abstract class defining an interface for
    other transforms.
    """

    def __init__(self, rng=None):

        if rng is None:
            seed = [2015, 10, 1]
            self.rng = numpy.random.RandomState(seed)
        else:
            # print "inherited"
            self.rng = rng

    def fprop(self, inputs):
        """
        Implements a forward propagation through the i-th layer, that is
        some form of:
           $a^{i} = xW^{i} + b^{i}$
           $h^{i} = f^{i}(a^{i})$
        with f^{i}, W^{i}, b^{i} denoting a non-linearity, weight matrix and
        biases at the i-th layer, respectively and x denoting inputs.

        :param inputs: matrix of features $(x)$ or the output of the previous layer $h^{i-1}$
        :return: $h^i$, matrix of transformed by layer features
        """
        raise NotImplementedError()

    def bprop(self, h, igrads):
        """
        Implements a backward propagation through the layer, that is, given
        $h^i$ denotes the output of the layer and $x^i$ the input, we compute:
        $dh^i/dx^i$ which by chain rule is $dh^i/da^i da^i/dx^i$
        $x^i$ could be either features $(x)$ or the output of the lower layer $h^{i-1}$
        :param h: it's an activation produced in forward pass
        :param igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the $\partial h^{i}/ \partial a^{i}$ derivative
        :return: a tuple (deltas, ograds) where:
               $deltas = igrads *  \partial h^{i}/ \partial a^{i}$
               $ograds = deltas \times \partial a^{i}/ \partial x^{i}$
        """
        raise NotImplementedError()

    def bprop_cost(self, h, igrads, cost=None):
        """
        Implements a backward propagation in case the layer directly
        deals with the optimised cost (i.e. the top layer)
        By default, method should implement a back-prop for default cost, that is
        the one that is natural to the layer's output, i.e.:
        linear -> mse, softmax -> cross-entropy, sigmoid -> binary cross-entropy
        :param $h$: it's an activation produced in forward pass
        :param igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the $\partial h^{i}/ \partial a^{i}$ derivative
        :return: a tuple (deltas, ograds) where:
               $deltas = igrads * \partial h^{i}/ \partial a^{i}$
               $ograds = deltas \times \partial a^{i}/ \partial x^{i}$
        """

        raise NotImplementedError()

    def pgrads(self, inputs, deltas, **kwargs):
        """
        Return gradients w.r.t parameters
        """
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self):
        raise NotImplementedError()

    def get_name(self):
        return 'abstract_layer'


class Linear(Layer):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Linear, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim
        self.irange = irange
        self.W = self.rng.uniform(
            -irange, irange,
            (self.idim, self.odim))
        # print self.rng, irange
        # print self.W
        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)

    def fprop(self, inputs):
        """
        Implements a forward propagation through the i-th layer, that is
        some form of:
           $a^{i} = xW^{i} + b^{i}$
           $h^{i} = f^{i}(a^{i})$
        with $f^{i}, W^{i}, b^{i}$ denoting a non-linearity, weight matrix and
        biases of this (i-th) layer, respectively and $x$ denoting inputs.

        :param inputs: matrix of features $(x)$ or the output of the previous layer $h^{i-1}$
        :return: $h^i$, matrix of transformed by layer features
        """

        # input comes from 4D convolutional tensor, reshape to expected shape
        if inputs.ndim >2:
            inputs = inputs.reshape(inputs.shape[0], -1)

        a = numpy.dot(inputs, self.W) + self.b
        # here f() is an identity function, so just return a linear
        # transformation
        return a

    def bprop(self, h, igrads):
        """
        Implements a backward propagation through the layer, that is, given
        $h^i$ denotes the output of the layer and $x^i$ the input, we compute:
        $dh^i/dx^i$ which by chain rule is $dh^i/da^i da^i/dx^i$
        $x^i$ could be either features $(x)$ or the output of the lower layer $h^{i-1}$
        :param h: it's an activation produced in forward pass
        :param igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the $\partial h^{i}/ \partial a^{i}$ derivative
        :return: a tuple (deltas, ograds) where:
               $deltas = igrads *  \partial h^{i}/ \partial a^{i}$
               $ograds = deltas \times \partial a^{i}/ \partial x^{i}$
        """

        # since $df^i/da^i = 1$  (f is assumed identity function),
        # deltas are in fact the same as igrads
        # print igrads.shape
        if igrads.ndim >2:
            igrads = igrads.reshape(igrads.shape[0], -1)
        # print igrads.shape
        ograds = numpy.dot(igrads, self.W.T)
        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):
        """
        Implements a backward propagation in case the layer directly
        deals with the optimised cost (i.e. the top layer)
        By default, method should implement a back-prop for default cost, that is
        the one that is natural to the layer's output, i.e.:
        linear -> mse, softmax -> cross-entropy, sigmoid -> binary cross-entropy
        :param $h$: it's an activation produced in forward pass
        :param igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the $\partial h^{i}/ \partial a^{i}$ derivative
        :return: a tuple (deltas, ograds) where:
               $deltas = igrads * \partial h^{i}/ \partial a^{i}$
               $ograds = deltas \times \partial a^{i}/ \partial x^{i}$
        """

        if cost is None or cost.get_name() == 'mse':
            # for linear layer and mean square error cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Linear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        """
        Return gradients w.r.t parameters

        :param inputs, input to the i-th layer
        :param deltas, deltas computed in bprop stage up to -ith layer
        :param kwargs, key-value optional arguments
        :return list of grads w.r.t parameters dE/dW and dE/db in *exactly*
                the same order as the params are returned by get_params()

        Note: deltas here contain the whole chain rule leading
        from the cost up to the the i-th layer, i.e.
        $\partial E/\partial y^{L}  \partial y^{L}/ \partial a^{L} \partial a^{L}/ \partial h^{L-1} \partial h^{L-1}/ \partial a^{L-1} ... \partial h^{i}/ \partial a^{i}$
        and here we are just asking about
          1) $\partial a^{i}/ \partial W^{i}$ and 2) $\partial a^{i}/ \partial b^{i}$
        since W and b are only layer's parameters
        """

        # input comes from 4D convolutional tensor, reshape to expected shape
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)

        # you could basically use different scalers for biases
        # and weights, but it is not implemented here like this
        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight*self.W
            l2_b_penalty = l2_weight*self.b

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight*numpy.sign(self.W)
            l1_b_penalty = l1_weight*numpy.sign(self.b)

        grad_W = numpy.dot(inputs.T, deltas) + l2_W_penalty + l1_W_penalty
        grad_b = numpy.sum(deltas, axis=0) + l2_b_penalty + l1_b_penalty

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        # we do not make checks here, but the order on the list
        # is assumed to be exactly the same as get_params() returns
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'linear'


class Sigmoid(Linear):

    def __init__(self,  idim, odim,
                 rng=None,
                 irange=0.1):

        super(Sigmoid, self).__init__(idim, odim, rng, irange)

    def fprop(self, inputs):
        # get the linear activations
        a = super(Sigmoid, self).fprop(inputs)
        # stabilise the exp() computation in case some values in
        #'a' get very negative. We limit both tails, however only
        # negative values may lead to numerical issues -- $e^{-a}$
        # clip() function does the following operation faster:
        # a[a < -30.] = -30,
        # a[a > 30.] = 30.
        numpy.clip(a, -30.0, 30.0, out=a)
        h = 1.0/(1 + numpy.exp(-a))
        return h

    def bprop(self, h, igrads):

        if igrads.ndim >2:
            igrads = igrads.reshape(igrads.shape[0], -1)
        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(Sigmoid, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'bce' or True:
            return super(Sigmoid, self).bprop(h=h, igrads=igrads)
        else:
            raise NotImplementedError('Sigmoid.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'sigmoid'


class Softmax(Linear):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Softmax, self).__init__(idim,
                                      odim,
                                      rng=rng,
                                      irange=irange)

    def fprop(self, inputs):

        # compute the linear outputs
        a = super(Softmax, self).fprop(inputs)
        # apply numerical stabilisation by subtracting max
        # from each row (not required for the coursework)
        # then compute exponent
        assert a.ndim in [1, 2], (
            "Expected the linear activation in Softmax layer to be either "
            "vector or matrix, got %ith dimensional tensor" % a.ndim
        )
        axis = a.ndim - 1
        exp_a = numpy.exp(a - numpy.max(a, axis=axis, keepdims=True))
        # finally, normalise by the sum within each example
        y = exp_a/numpy.sum(exp_a, axis=axis, keepdims=True)

        return y

    def bprop(self, h, igrads):
        raise NotImplementedError(
            'Softmax.bprop not implemented for hidden layer.')

    def bprop_cost(self, h, igrads, cost):

        if cost is None or cost.get_name() == 'ce':
            return super(Softmax, self).bprop(h=h, igrads=igrads)
        else:
            raise NotImplementedError('Softmax.bprop_cost method not implemented '
                                      'for %s cost' % cost.get_name())

    def get_name(self):
        return 'softmax'


class Relu(Linear):

    def __init__(self,  idim, odim,
                 rng=None,
                 irange=0.1):

        super(Relu, self).__init__(idim, odim, rng, irange)

    def fprop(self, inputs):
        # get the linear activations
        a = super(Relu, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        #h = numpy.maximum(a, 0)
        return h

    def bprop(self, h, igrads):
        deltas = (h > 0)*igrads
        ___, ograds = super(Relu, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Relu.bprop_cost method not implemented '
                                  'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'relu'


class Tanh(Linear):

    def __init__(self,  idim, odim,
                 rng=None,
                 irange=0.1):

        super(Tanh, self).__init__(idim, odim, rng, irange)

    def fprop(self, inputs):
        # get the linear activations
        a = super(Tanh, self).fprop(inputs)
        numpy.clip(a, -30.0, 30.0, out=a)
        h = numpy.tanh(a)
        return h

    def bprop(self, h, igrads):
        deltas = (1.0 - h**2) * igrads
        ___, ograds = super(Tanh, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Tanh.bprop_cost method not implemented '
                                  'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'tanh'


class Maxout(Linear):

    def __init__(self,  idim, odim, k,
                 rng=None,
                 irange=0.05):

        super(Maxout, self).__init__(idim, odim*k, rng, irange)

        self.max_odim = odim
        self.k = k

    def fprop(self, inputs):
        # get the linear activations
        a = super(Maxout, self).fprop(inputs)
        ar = a.reshape(a.shape[0], self.max_odim, self.k)
        h, h_argmax = max_and_argmax(
            ar, axes=2, keepdims_max=True, keepdims_argmax=True)
        self.h_argmax = h_argmax
        return h[:, :, 0]  # get rid of the last reduced dimensison (of size 1)

    def bprop(self, h, igrads):
        # hack for dropout backprop (ignore dropped neurons). Note, this is not
        # entirely correct when h fires at 0 exaclty (but is not dropped, in which case
        # derivative should be 1). However, this is rather unlikely to happen (that h fires as 0)
        # and probably can be ignored for now. Otherwise, one would have to keep the dropped unit
        # indexes and zero grads according to them.
        igrads = (h != 0)*igrads
        # convert into the shape where upsampling is easier
        igrads_up = igrads.reshape(igrads.shape[0], self.max_odim, 1)
        # upsample to the linear dimension (but reshaped to (batch_size,
        # maxed_num (1), pool_size)
        igrads_up = numpy.tile(igrads_up, (1, 1, self.k))
        # generate mask matrix and set to 1 maxed elements
        mask = numpy.zeros_like(igrads_up)
        mask[self.h_argmax] = 1.0
        # do bprop through max operator and then reshape into 2D
        deltas = (igrads_up * mask).reshape(igrads_up.shape[0], -1)
        # and then do bprop thorough linear part
        ___, ograds = super(Maxout, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Maxout.bprop_cost method not implemented '
                                  'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'maxout'


class ComplexLinear(Layer):
    """
    This layer sets up the two weight matrices required for the ADFT layer
    implemented as compelx abs
    """

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(ComplexLinear, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim
        # must be true for fourier layer
        assert idim == odim

        # odim  is output and the input dimensions of the DFT layer
        dft_mat = dft(odim).T
        # diag(M * M^{dagger}) gives me the normalization values for each row
        # setting M = M^T does the trick for collumns
        normalization_vals = numpy.diag((dft_mat.T).dot(numpy.conj(dft_mat.T)))
        # This is the renormalized weight matrix from my understanding
        self.W = dft_mat  # / normalization_vals.reshape(-1,1)
        # seperating in to two (outputing double number of neurons )
        self.Wr = numpy.real(self.W)
        self.Wi = numpy.imag(self.W)

        self.b = numpy.zeros((self.odim*2,), dtype=numpy.float32)

    def norm_weights(self, train_iterator):
        """
        Normalize dft weights by $E[(wx)^2]$
        """
        n = 1
        for inputs, t in train_iterator:
            if n == 1:
                # + self.b
                n1 = (1.0/inputs.shape[1])*numpy.dot(inputs**2, self.Wr**2)
                n2 = (1.0/inputs.shape[1])*numpy.dot(inputs**2, self.Wi**2)
                n += 1
                continue
            n1 += (1.0/inputs.shape[1]) * \
                numpy.dot(inputs**2, self.Wr**2)  # + self.b
            n2 += (1.0/inputs.shape[1])*numpy.dot(inputs**2, self.Wi**2)
            n += 1
        n1 /= n
        n2 /= n
        self.Wr = self.Wr / numpy.mean(n1, axis=0)
        n2m = numpy.mean(n2, axis=0)
        n2m[n2m == 0.0] = 1.0
        self.Wi = self.Wi / n2m

    def fprop(self, inputs):

        # input comes from 4D convolutional tensor, reshape to expected shape
        # a_old = numpy.dot(inputs, self.W)

        a1 = numpy.dot(inputs, self.Wr)  # + self.b
        a2 = numpy.dot(inputs, self.Wi)  # + self.b
        a = numpy.concatenate((a1, a2), axis=1)
        # a = a_old + self.b
        # here f() is an identity function, so just return a linear
        # transformation
        return a + self.b

    def bprop(self, h, igrads):
        # self.first = False
        # since df^i/da^i = 1 (f is assumed identity function),
        # deltas are in fact the same as igrads
        igrads = igrads.reshape(igrads.shape[0], 2,
                                igrads.shape[1] / 2)
        ir = numpy.dot(igrads[:, ::2, :], self.Wr.T)
        ii = numpy.dot(igrads[:, 1::2, :], self.Wi.T)
        ograds = numpy.concatenate((ir, ii), axis=1)
        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):

        if cost is None or cost.get_name() == 'mse':
            # for linear layer and mean square error cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Linear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())


    def get_params(self):
        return [self.Wr, self.Wi, self.b]

    def set_params(self, params):
        # we do not make checks here, but the order on the list
        # is assumed to be exactly the same as get_params() returns
        self.Wr = params[0]
        self.Wi = params[1]
        self.b = params[2]

    def get_name(self):
        return 'clinear'

class ComplexAbs(ComplexLinear):

    """
    Implements an ADFT (Spectral layer) with $|a|^{2}$ based
    activation functions.
    """

    def __init__(self,  idim, odim,
                 rng=None,
                 irange=0.1):

        super(ComplexAbs, self).__init__(idim, odim, rng, irange)

    def fprop(self, inputs):

        # input comes from 4D convolutional tensor, reshape to expected shape

        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)

        a1 = numpy.power(numpy.dot(inputs, self.Wr), 2)  # + self.b
        a2 = numpy.power(numpy.dot(inputs, self.Wi), 2)  # + self.b

        return numpy.sqrt(a1 + a2)

    def bprop(self, h, igrads):

        if igrads.ndim >2:
            igrads = igrads.reshape(igrads.shape[0], -1)

        # Derivative of $h(a)= \sqrt{a}$ is $h(a)^{-1}$
        igrads = igrads / h

        # for backpropagating to next layer (ograds)
        ir = numpy.dot(igrads, self.Wr.T)
        ii = numpy.dot(igrads, self.Wi.T)

        return igrads, ir + ii

    def bprop_cost(self, h, igrads, cost):

        raise NotImplementedError('Sigmoid.bprop_cost method not implemented '
                                  'for the %s cost' % cost.get_name())

    def pgrads(self, inputs, deltas,
               l1_weight=0, l2_weight=0):

        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)

        grad_Wr = numpy.dot(inputs.T,
                            numpy.dot(inputs, self.Wr) * deltas)
        grad_Wi = numpy.dot(inputs.T,
                            numpy.dot(inputs, self.Wi) * deltas)

        return [grad_Wr, grad_Wi, 0]


    def get_name(self):
        return 'abs'
