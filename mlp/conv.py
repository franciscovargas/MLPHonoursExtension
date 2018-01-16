#	//check that pattern is not longer than the tex://github.com/franciscovargas/MLPHonoursExtension.githttps://github.com/franciscovargas/MLPHonoursExtension.githttps://github.com/franciscovargas/MLPHonoursExtension.githttps://github.com/franciscovargas/MLPHonoursExtension.git Francisco Vargas
# TODO: rename vars like woo ...
import numpy
import logging
from mlp.layers import Layer, Linear
import numpy as np
from mlp.layers import max_and_argmax
from numpy.lib import stride_tricks as st
from copy import deepcopy



logger = logging.getLogger(__name__)


"""
"In that blessed region of Four Dimensions,
shall we linger on the threshold of the Fifth, and not enter therein?"

- Flatland, Edwin Abbott.

Ready your swords here be tensors.
"""

def my1_conv2d_tense(image, kernels, mode='fprop', strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those
    with the input spaceproduces feature maps (sometimes refereed to also as receptive
    fields). Also note, that feature maps are synonyms here to channels, and as such
    num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """

    if (mode=='fprop'):

        return strided_convt(image, kernels, stridz=strides)
    else:
        I_n = image.shape[0]
        I_x = image.shape[1]
        I_y = image.shape[2]
        k_x = kernels.shape[-2]
        k_y = kernels.shape[-1]
        image = np.pad(image,
                       ((0,0),(0,0),(k_x-1, k_x  -1),
                       (k_y-1, k_y -1)),
                       'constant',
                       constant_values=0 )
        return strided_convt(image, kernels, mode='bprop')



def strided_convt(I, k, stridz=(1,1), mode='fprop'):
    """
    matrix mult version tensor 1 ver
    k tensor I tensor
    """
    
    b1, nf_i, h, w = I.shape
    b,i, k_y, k_x = k.shape
    ystep , xstep = stridz
    nh, nw =((abs(h - k_y) + 1) / ystep, (abs(w - k_x) + 1) / xstep)

    # This stride bassically generates a permutation
    # matrix which allows to do the convolution operation in a fully
    # vectorized form. This also allows automatic striding.
    windows = st.as_strided(I, (b1, nf_i, nh, nw, k_y, k_x),
     (I.strides[-4],I.strides[-3], I.strides[-2] * ystep,
      I.strides[-1] * xstep, I.strides[-2], I.strides[-1]))

    woo = windows.reshape(b1, nf_i, nh, nw, k_y* k_x)

    koo = k.reshape(b, i, -1)
    print woo.shape, koo.shape
    # Einsteins summation convention allows us to map dot products conviently
    # within tensors in a loop like fashion
    dt = np.einsum('ijklm,jim->ijkl', woo, koo)
    return dt


class ConvLinear_Opt(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d_tense,
                 conv_bck=my1_conv2d_tense,
                 conv_grad=my1_conv2d_tense):
        """
        Almost fully vectorized linear convolution:

        --- fprop + fprop_convolution: 1 loop
        --- bprop + bprop_convolution: 1 loops
        --- pgrads + pgrads_convolution: 2 loops

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
        self.stride = stride
        self.odim = odim
        self.kernel_shape = kernel_shape
        self.image_shape = image_shape
        self.num_inp_feat_maps = num_inp_feat_maps
        self.num_out_feat_maps = num_out_feat_maps
        self.irange = irange
        self.W = self.weight()
        self.b = numpy.zeros((1, self.num_out_feat_maps,odim,
                              odim),
                             dtype=numpy.float32)
        if self.stride[0] > 1:
            self.b = numpy.zeros((1, self.num_out_feat_maps,odim/stride[0],
                                  odim/stride[1]),
                                 dtype=numpy.float32)



    def weight(self):
        return self.rng.uniform(
                    -self.irange, self.irange,
                    (self.num_inp_feat_maps,  self.num_out_feat_maps,
                     self.kernel_shape[0], self.kernel_shape[1]))


    def fprop(self, inputs):

        image_shape = self.image_shape
        kernel_shape = self.kernel_shape
        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = inputs.shape[0]
        n_chan = inputs.shape[1]
        sy, sx = self.stride


        out = np.zeros((inputs.shape[0], f_out,
                        image_shape[0] - kernel_shape[0] + 1,
                        image_shape[1] - kernel_shape[1] + 1))

        # Reshaping of outputs due to strides conditions
        if self.stride[0] > 1:
            out = np.zeros((inputs.shape[0], f_out,
                           (image_shape[0] - kernel_shape[0] + 1) / sy,
                            (image_shape[1] - kernel_shape[1] + 1 ) /  sx))

        x = inputs
        ww = self.W

        # My vectorization left only the channels out loop
        for k in xrange(f_out):
            w = ww[:,k,:,:] \
                .repeat(x.shape[0], axis=0) \
                .reshape(f_in, -1, kernel_shape[-2], kernel_shape[-1])

            out[:,k,:,:] = np.sum(my1_conv2d_tense(x, w, strides = self.stride,
                                                   mode='fprop'), axis=1)
            out[:,k,:,:] += self.b[0,k,:,:]

        return out

    def bprop(self, h, igrads):

        image_shape = self.image_shape
        kernel_shape = self.kernel_shape

        
        igrads  = igrads.reshape(igrads.shape[0],
                                 self.num_out_feat_maps,
                                 self.odim, self.odim)

        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = igrads.shape[0]
        n_chan = igrads.shape[1]

        ograds = np.zeros((igrads.shape[0], f_in, image_shape[0] ,
                           image_shape[1] ))


        # 180 rotation of the weight matrix
        ww = self.W[...,::-1,::-1]
        ww = ww.swapaxes(0,1)

        # Again only channels out loop post vectorization
        for k in xrange(f_in):
            w = ww[:,k,:,:] \
                  .repeat(igrads.shape[0], axis=0) \
                  .reshape(f_out, -1, kernel_shape[-2], kernel_shape[-1])            
            ograds[:,k,:,:]= my1_conv2d_tense(igrads, w, mode='bprop').sum(axis=1)
        return igrads, ograds

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

        f_in = self.W.shape[0]
        f_out = self.W.shape[1]
        batch_n = deltas.shape[0]
        n_chan = deltas.shape[1]

        grad_W = np.zeros((f_in, f_out, self.W.shape[2] ,
                          self.W.shape[3] ))
 
        deltas = deltas.swapaxes(0,1)
     
        for k in xrange(f_out):
            w = deltas[k,:,:,:]\
                    .reshape(1, deltas.shape[1] * deltas.shape[2], deltas.shape[3])\
                    .repeat(inputs.shape[1], axis=0) \
                    .reshape(inputs.shape[1],-1, deltas.shape[-2], deltas.shape[-1])
            grad_W[:,k,:,:] = my1_conv2d_tense(inputs, w).sum(axis=0)
  
        deltas = deltas.swapaxes(0,1)

        grad_b = numpy.sum(deltas, axis=0)

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):

        # For test purposes
        if params[1].ndim == 1:
            self.W = params[0]
            self.b = params[1].repeat(self.odim*self.odim)\
                     .reshape(1, self.num_out_feat_maps,
                              self.odim, self.odim)
        else:
            self.W = params[0]
            self.b = params[1]

    def get_name(self):
        return 'convlinear'


class ConvSigmoid_Opt(ConvLinear_Opt):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d_tense,
                 conv_bck=my1_conv2d_tense,
                 conv_grad=my1_conv2d_tense):
        super(ConvSigmoid_Opt, self).__init__(num_inp_feat_maps,
                                              num_out_feat_maps,
                                              image_shape=image_shape,
                                              kernel_shape=kernel_shape,
                                              stride=stride,
                                              irange=irange,
                                              rng=rng,
                                              conv_fwd=my1_conv2d_tense,
                                              conv_bck=my1_conv2d_tense,
                                              conv_grad=my1_conv2d_tense)


    def fprop(self, inputs):

        a = super(ConvSigmoid_Opt, self).fprop(inputs)

        numpy.clip(a, -30.0, 30.0, out=a)
        h = 1.0/(1 + numpy.exp(-a))
        return h

    def bprop(self, h, igrads):

        if self.stride[0] > 1:
                igrads  = igrads.reshape(igrads.shape[0],
                                         self.num_out_feat_maps,
                                         self.odim/self.stride[0],
                                         self.odim/self.stride[1])
        else:
            igrads  = igrads.reshape(igrads.shape[0],
                                     self.num_out_feat_maps,
                                     self.odim, self.odim)
        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(ConvSigmoid_Opt, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'bce':
            return super(ConvSigmoid_Opt, self).bprop(h=h, igrads=igrads)
        else:
            raise NotImplementedError('Sigmoid.bprop_cost method not implemented'
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'ConvSigmoid'


class ConvRelu_Opt(ConvLinear_Opt):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d_tense,
                 conv_bck=my1_conv2d_tense,
                 conv_grad=my1_conv2d_tense):
        super(ConvRelu_Opt, self).__init__(num_inp_feat_maps,
                                           num_out_feat_maps,
                                           image_shape=image_shape,
                                           kernel_shape=kernel_shape,
                                           stride=stride,
                                           irange=irange,
                                           rng=rng,
                                           conv_fwd=my1_conv2d_tense,
                                           conv_bck=my1_conv2d_tense,
                                           conv_grad=my1_conv2d_tense)


    def fprop(self, inputs):
        # Get the linear activations
        a = super(ConvRelu_Opt, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        h = numpy.maximum(a, 0)
        return h

    def bprop(self, h, igrads):
        if self.stride[0] > 1:
                igrads  = igrads.reshape(igrads.shape[0],
                                         self.num_out_feat_maps,
                                         self.odim/self.stride[0],
                                         self.odim/self.stride[1])
        else:
            igrads  = igrads.reshape(igrads.shape[0],
                                     self.num_out_feat_maps,
                                     self.odim, self.odim)

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
                 pool_shape=(2, 2),
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
        self.p_y, self.p_x = pool_shape
        self.pool_stride = pool_stride


    @staticmethod
    def vectorized_Gneration(a,indices):
        """
        Vectorized index bijection for 4D tensor must work on 5D
        5D it is must test.
        """
        l,p,m,n,r = a.shape
        a.reshape(-1,r)[np.arange(l*p*m*n),indices.ravel()] = 1

    def fprop(self, inputs):
        """
        "In that blessed region of Four Dimensions,
        shall we linger on the threshold of the Fifth, and not enter therein?"

        - Flatland, Edwin Abbott.
        """
        p_y, p_x = self.pool_shape
        N, nf_i, h, w = inputs.shape

        N, nf_i, nh, nw = (N, nf_i, h/p_y, w/p_x)

        arg = inputs.reshape(N, nf_i, nh, p_y, nw, p_x).swapaxes(3,4) .reshape(N, nf_i,-1, p_x*p_y )

        O, I =  max_and_argmax(arg, axes=3)

        self.G = np.zeros((N, nf_i, nh, nw, p_x*p_y))
        self.vectorized_Gneration(self.G, I)
        self.G = self.G.reshape(N, nf_i, nh, nw,p_y,p_x).swapaxes(3,4).reshape(N, nf_i, h, w)


        return O.reshape(N, nf_i, nh, nw)


    def bprop(self, h, igrads):

        igrads = igrads.reshape(self.conv_shape[0],
                                self.num_feat_maps,
                                self.conv_shape[-2] / self.p_x,
                                self.conv_shape[-1] / self.p_y)
        ograds = self.G * igrads.repeat(self.pool_shape[0], 2)\
                                .repeat(self.pool_shape[1], 3)

        return igrads, ograds

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'

    
 
class ConvMaxPool2DStrides(Layer):
    # To be set by fprop
    G = None

    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional
               feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        self.num_feat_maps = num_feat_maps
        self.conv_shape = conv_shape
        self.pool_shape = pool_shape
        self.p_y, self.p_x = pool_shape
        self.pool_stride = pool_stride
        self.stride = pool_stride
        self.s_y, self.s_x = pool_stride


    @staticmethod
    def vectorized_Gneration(a,indices):
        """
        Vectorized index bijection for 5D tensors
        Used in maxpooling for the construction of matrix
        G
        """
        l,p,m,n,r = a.shape
        a.reshape(-1,r)[np.arange(l*p*m*n),indices.ravel()] = 1

    def fprop(self, inputs):

        s_y, s_x = self.stride
        p_y, p_x = self.pool_shape
        N, nf_i, h, w = inputs.shape
        # Safety Hacks:
        input_tmp = deepcopy(inputs)


        # The non overlapping stride is implemented by setting
        # the rows and collumns to skip to 0 and expanding the kernel
        # in such way that they are covered (but do not contribute)
        # Addtionally the image is also expanded such that it
        # fits the new kernel size. This was my vectorized way
        #  of inplementing the strides
        # This is all carried out under the assumption that the kernel is symmetric
        # since otherwise I would have a bunch of even more ugglier condtionals
        # print p_y, p_x
        # good accuracies and tested results without strides I ran out of time
        # for the strides
        if self.stride[0] > 2:

            # Resizing the kernel
            p_y += (s_y - 2)
            p_x += (s_x - 2)

            # Calculation of outer image padding
            x_pad = (w % p_x)
            y_pad = (h % p_y)

            # Outer image padding for larger kernel
            input_tmp = np.pad(input_tmp,
                               ((0, 0),
                                (0, 0),
                                (0, y_pad),
                                (0, x_pad)),
                               'constant',
                               constant_values=0)
            input_tmp[:,:,p_y:-1:p_y,:] = 0
            input_tmp[:,:,:,p_y:-1:p_y] = 0

        N, nf_i, nh, nw = (N, nf_i, h/p_y, w/p_x)


        arg = input_tmp.reshape(N, nf_i, nh, p_y, nw, p_x).swapaxes(3,4)\
                       .reshape(N, nf_i,nh*nw, p_x*p_y )

        O, I =  max_and_argmax(arg, axes=3)

        # Construction of matrix G and unlike the lectures it is applied via
        # Hadamard elementwise multiplication instead of matrix mult
        self.G = np.zeros((N, nf_i, nh, nw, p_x*p_y))
        self.vectorized_Gneration(self.G, I)
        self.G = self.G.reshape(N, nf_i, nh, nw,p_y,p_x).swapaxes(3,4)\
                       .reshape(N, nf_i, h, w)


        return O.reshape(N, nf_i, nh, nw)

    def bprop(self, h, igrads):

        # little hacks for strides
        # such that dimensions align when Reshaping
        # it is possible  that my matrix G is no longer correct
        # when using strides but results are not so bad so I think its ok
        # Reaches 97% on small layers with stride (3,3)
        if self.s_x <= 2:
            s_x = 0
            s_y = 0
            f = 0
        else:
            s_x = self.s_x
            s_y = self.s_y
            f = 2


        igrads = igrads.reshape(self.conv_shape[0] , self.num_feat_maps,
                                self.conv_shape[-2]/(self.p_y+s_y-f),
                                self.conv_shape[-1]/(self.p_x+s_x-f))

        ograds = self.G * igrads.repeat(self.pool_shape[0] +s_x-f, 2)\
                                .repeat(self.pool_shape[1] +s_x-f, 3)

        return igrads, ograds

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'
