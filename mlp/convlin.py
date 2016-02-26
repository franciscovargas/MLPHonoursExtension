# %load mlp/conv.py

# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

# Messy and very hardcoded
import numpy
import logging
from mlp.layers import Layer, Linear
from scipy.signal import convolve, correlate
import numpy as np
from mlp.layers import max_and_argmax


logger = logging.getLogger(__name__)

"""
You have been given some very initial skeleton below. Feel free to build on top of it and/or
modify it according to your needs. Just notice, you can factor out the convolution code out of
the layer code, and just pass (possibly) different conv implementations for each of the stages
in the model where you are expected to apply the convolutional operator. This will allow you to
keep the layer implementation independent of conv operator implementation, and you can easily
swap it layer, for example, for more efficient implementation if you came up with one, etc.
"""

def my1_conv2d(image, kernels, mode='fprop', strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """

    if (mode=='fprop'):

        return myconv(image, kernels)
    else:
        I_x = image.shape[0]
        k_x = kernels.shape[0]

        image = np.pad(image,
                       ((k_x-1, k_x  -1)),
                       'constant',
                       constant_values=0 )
        return myconv(image, kernels)


def my1_conv2d_tense(image, kernels, mode='fprop', strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """

    if (mode=='fprop'):

        return myconv_tense(image, kernels)
    else:
        I_n = image.shape[0]
        I_x = image.shape[1]
        k_x = kernels.shape[0]

        image = np.pad(image,
                       ((0,0),(k_x-1, k_x  -1)),
                       'constant',
                       constant_values=0 )
        return myconv_tense(image, kernels)


def my1_conv2d_tense2(image, kernels, mode='fprop', strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """

    if (mode=='fprop'):

        return myconv_tense2(image, kernels)
    else:
        I_n = image.shape[0]
        I_x = image.shape[1]
        k_x = kernels.shape[0]

        image = np.pad(image,
                       ((0,0),(k_x-1, k_x  -1)),
                       'constant',
                       constant_values=0 )
        return myconv_tense2(image, kernels)


def myconv(I,k):
    I_x = I.shape[0]

    k_x = k.shape[0]

    out = np.zeros((I_x - k_x + 1))
    o_y = out.shape[-1]

    for y in xrange(k_x):
        # for x in xrange(k_x):
        out += k[y] * I[y:o_y+y]
    return out


def myconv_tense(I,k):
    """
    Tensor vectorized  convolution calculation for bprop and fprop
    I is  a 3D tensor k is not
    """
    I_n = I.shape[0]
    I_x = I.shape[1]
    # I_y = I.shape[2]


    # k_x = k.shape[0]
    k_x = k.shape[0]
    out = np.zeros((I_n, I_x - k_x + 1))
    o_y = out.shape[-1]


    # k is smaller thus looping over k brings a
    # great speed overhead
    for y in xrange(k_x):
        # for x in xrange(k_x):
        out += k[y] * I[:,y:o_y+y]

    return out

def myconv_tense2(I,k):
    """
    Tensor vectorized convolution calculation for pgrads
    Both I and k are 3D tensors
    """
    I_n = I.shape[0]
    # print I.shape[0]
    I_x = I.shape[1]

    k_x = k.shape[1]
    out = np.zeros((I_n, I_x - k_x + 1))
    o_y = out.shape[-1]


    for y in xrange(k_x):
        # for x in xrange(k_x):
        out += k[:, y].reshape(-1,1) * I[:,y:o_y+y]
    # print out.shape
    return out


class ConvLinear_Opt(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(125,),
                 kernel_shape=(4, 4),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvLinear_Opt, self).__init__(rng=rng)

        odim = (image_shape[0] - kernel_shape[0] + 1)
        self.odim = odim
        self.kernel_shape = kernel_shape
        self.image_shape = image_shape
        self.num_inp_feat_maps = num_inp_feat_maps
        self.num_out_feat_maps = num_out_feat_maps
        self.irange = irange
        self.W = self.weight()
        self.b = numpy.zeros((1, self.num_out_feat_maps,odim),
                             dtype=numpy.float32)


    def weight(self):
        return self.rng.uniform(
                    -self.irange, self.irange,
                    (self.num_inp_feat_maps,  self.num_out_feat_maps,
                     self.kernel_shape[0]))

    def fprop(self, inputs):

        image_shape = self.image_shape
        kernel_shape = self.kernel_shape
        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = inputs.shape[0]
        n_chan = inputs.shape[1]

        out = np.zeros((inputs.shape[0], f_out, image_shape[0] - kernel_shape[0] + 1))


        x = inputs.swapaxes(0,1)

        for k in xrange(f_out):
            for j in xrange(f_in):
                w = self.W[j,k,:]
                x_in = x[j,:]
                out[:,k,:] += my1_conv2d_tense(x_in, w, mode='fprop')
            out[:,k,:] += self.b[0,k,:]

        return out

    def bprop(self, h, igrads):

        image_shape = self.image_shape
        kernel_shape = self.kernel_shape

        igrads  = igrads.reshape(igrads.shape[0], self.num_out_feat_maps,self.odim)
        deltas = igrads

        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = igrads.shape[0]
        n_chan = igrads.shape[1]

        ograds = np.zeros((igrads.shape[0], f_in, image_shape[0]  ))

        x = igrads.swapaxes(0,1)
        for k in xrange(f_out):
            x_in = x[k,:,:]
            for j in xrange(f_in):
                w = self.W[j,k,::-1]
                conv =  my1_conv2d_tense(x_in, w, mode='bprop')
                ograds[:,j,:] += conv
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')


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
        dE/dy^L dy^L/da^L da^L/dh^{L-1} dh^{L-1}/da^{L-1} ... dh^{i}/da^{i}
        and here we are just asking about
          1) da^i/dW^i and 2) da^i/db^i
        since W and b are only layer's parameters
        """



        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight*self.W
            l2_b_penalty = l2_weight*self.b

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight*numpy.sign(self.W)
            l1_b_penalty = l1_weight*numpy.sign(self.b)

        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = deltas.shape[0]
        n_chan = deltas.shape[1]

        grad_W = np.zeros((f_in, f_out, self.W.shape[2] ))
        x = inputs.swapaxes(0,1)
        deltas = deltas.swapaxes(0,1)
        for j in xrange(f_in):
            x_in = x[j,:,:]
            for k in xrange(f_out):
                dc = deltas[k,:,:]
                grad_W[j,k,:] += np.sum(my1_conv2d_tense2(x_in, dc),axis=0) + l2_W_penalty + l1_W_penalty

        deltas = deltas.swapaxes(0,1)

        grad_b = numpy.sum(deltas, axis=0) + l2_b_penalty + l1_b_penalty


        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):

        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'


class ConvSigmoid_Opt(ConvLinear_Opt):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(125,),
                 kernel_shape=(4, 4),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvSigmoid_Opt, self).__init__(num_inp_feat_maps,
                                              num_out_feat_maps,
                                              image_shape=image_shape,
                                              kernel_shape=kernel_shape,
                                              stride=stride,
                                              irange=irange,
                                              rng=rng,
                                              conv_fwd=my1_conv2d,
                                              conv_bck=my1_conv2d,
                                              conv_grad=my1_conv2d)


    def fprop(self, inputs):

        a = super(ConvSigmoid_Opt, self).fprop(inputs)

        numpy.clip(a, -30.0, 30.0, out=a)
        h = 1.0/(1 + numpy.exp(-a))
        return h

    def bprop(self, h, igrads):
        igrads = igrads.reshape(igrads.shape[0], self.num_out_feat_maps,self.odim,self.odim)

        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(ConvSigmoid_Opt, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'bce':
            return super(ConvSigmoid_Opt, self).bprop(h=h, igrads=igrads)
        else:
            raise NotImplementedError('Sigmoid.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'ConvSigmoid'


class ConvRelu_Opt(ConvLinear_Opt):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(125,),
                 kernel_shape=(4, 4),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):

        super(ConvRelu_Opt, self).__init__(num_inp_feat_maps,
                                           num_out_feat_maps,
                                           image_shape=image_shape,
                                           kernel_shape=kernel_shape,
                                           stride=stride,
                                           irange=irange,
                                           rng=rng,
                                           conv_fwd=my1_conv2d,
                                           conv_bck=my1_conv2d,
                                           conv_grad=my1_conv2d)


    def fprop(self, inputs):
        #get the linear activations
        # print inputs.shape
        a = super(ConvRelu_Opt, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        h = numpy.maximum(a, 0)
        return h

    def bprop(self, h, igrads):
        igrads = igrads.reshape(igrads.shape[0], self.num_out_feat_maps,self.odim)

        deltas = (h > 0)*igrads
        ___, ograds = super(ConvRelu_Opt, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Relu.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'ConvRelu'





class ConvMaxPool2D(Layer):
    # To be set by fprop
    G = None

    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=2,
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        self.num_feat_maps = num_feat_maps
        self.conv_shape = conv_shape
        self.pool_shape = pool_shape
        self.p_x = pool_shape
        self.pool_stride = pool_stride


    @staticmethod
    def vectorized_Gneration(a,indices):
        """
        Vectorized index bijection for 4D tensor must work on 5D
        5D it is must test.
        """
        l,p,m,r = a.shape
        a.reshape(-1,r)[np.arange(l*p*m),indices.ravel()] = 1

    def fprop(self, inputs):
        """
        "In that blessed region of Four Dimensions,
        shall we linger on the threshold of the Fifth, and not enter therein?"

        - Flatland, Edwin Abbott.
        """
        p_x = self.pool_shape
        N, nf_i, h = inputs.shape

        N, nf_i, nh = (N, nf_i, h/p_x)

        arg = inputs.reshape(N, nf_i, nh, p_x)

        O = np.max(arg, axis=-1)
        I = np.argmax(arg, axis=-1)

        self.G = np.zeros((N, nf_i, nh, p_x))
        self.vectorized_Gneration(self.G, I)
        self.G = self.G.reshape(N, nf_i, h)


        return O.reshape(N, nf_i, nh)


    def bprop(self, h, igrads):

        igrads = igrads.reshape(100 , self.num_feat_maps, self.conv_shape[-1]/2)
        ograds = self.G * igrads.repeat(2, 2)

        return igrads, ograds

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'
