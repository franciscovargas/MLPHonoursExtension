
# Francisco Vargas

import cPickle
import gzip
import numpy
import os
from scipy.fftpack import fft as dft
import numpy as np
import logging
from scipy import signal



logger = logging.getLogger(__name__)


class DataProvider(object):
    """
    Data provider defines an interface for our
    generic data-independent readers.
    """
    def __init__(self, batch_size, randomize=True, rng=None):
        """
        :param batch_size: int, specifies the number
               of elements returned at each step
        :param randomize: bool, shuffles examples priorfprop
               to iteration, so they are presented in random
               order for stochastic gradient descent training
        :return:
        """
        self.batch_size = batch_size
        self.randomize = randomize
        self._curr_idx = 0
        self.rng = rng

        if self.rng is None:
            seed=[2015, 10, 1]
            self.rng = numpy.random.RandomState(seed)

    def reset(self):
        """
        Resets the provider to the initial state to
        use in another epoch
        :return: None
        """
        self._curr_idx = 0

    def __randomize(self):
        """
        Data-specific implementation of shuffling mechanism
        :return:
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        This method says an object is iterable.
        """
        return self

    def next(self):
        """
        Data-specific iteration mechanism. Called each step
        (i.e. each iteration in a loop)
        unitl StopIteration() exception is raised.
        :return:
        """
        raise NotImplementedError()

    def num_examples(self):
        """
        Returns a number of data-points in dataset
        """
        return NotImplementedError()


class mACLEnum(object):
    # http://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities
    # *3 is the z axis, no 11 is y-axis *3 I dont know
    #  3 ????????
    # Accelerometer dic (I hope... ffs)
    dic = {
            'LHAx': (18),
            'LLAx': (36),
            'RHAx': (9),
            'RLAx': (27),
            'CAx' : (0),
            'LHAy': (19),
            'LLAy': (37),
            'RHAy': (10),
            'RLAy': (28),
            'CAy' : (1),
            'LHAz': (20),
            'LLAz': (38),
            'RHAz': (11),
            'RLAz': (29),
            'CAz' : (2),

    }

    def __init__(self, name='RHAy'):
        self.s = self.dic[name]



class MACLDataProvider(DataProvider):
    """
    The class iterates over thingy action timeseries
    accelerometer based dataset
    """
    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True,
                 rng=None,
                 conv_reshape=False,name='RHAx', fft= False, stft=False):

        super(MACLDataProvider, self).\
            __init__(batch_size, randomize, rng)

        assert dset in ['train', 'valid', 'test'], (
            "Expected dset to be either 'train', "
            "'valid' or 'eval' got %s" % dset
        )

        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )

        if max_num_batches > 0 and max_num_examples > 0:
            logger.warning("You have specified both 'max_num_batches' and " \
                  "a deprecead 'max_num_examples' arguments. We will " \
                  "use the former over the latter.")

        dset_path = '/afs/inf.ed.ac.uk/user/s12/s1235260/Downloads/ACL1_%sv.pkl.gz' % dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        with gzip.open(dset_path) as f:
            x, t = cPickle.load(f)
            print x.shape , t.shape

        self._max_num_batches = max_num_batches
        #max_num_examples arg was provided for backward compatibility
        #but it maps us to the max_num_batches anyway
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size
        if isinstance(name, str):
            enum = mACLEnum(name)
            self.x = x[:,enum.s]
            if fft:
                self.x = np.abs(dft(self.x, axis=-1))
            if stft:
                # wind = signal.get_window('hamming', 7)
                f, t2, s = signal. spectrogram(self.x,
                                               nperseg=50,
                                               noverlap=50 - 25,
                                               axis=-1,
                                               scaling='spectrum',
                                               mode='magnitude')

                self.x = s.reshape(self.x.shape[0], -1)
        else:
            n = name.pop(0)
            enum = mACLEnum(n)
            if fft:
                self.x = np.abs(dft(x[:,enum.s], axis=-1))
            else:
                self.x = x[:,enum.s]
            for n in name:
                enum = mACLEnum(n)
                if fft:
                    self.x = numpy.concatenate((self.x ,
                                               np.abs(dft(x[:,enum.s],
                                                      axis=-1))), axis=-1)
                else:
                    self.x = numpy.concatenate((self.x , x[:, enum.s]), axis=-1)
            print "shape final: ", self.x.shape



            # print self.x.shape
        print self.x.shape
        self.t = t -1
        self.num_classes = 19
        self.conv_reshape = conv_reshape

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MACLDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)

        if self._rand_idx is not None and self._max_num_batches > 0:
            return self.rng.permutation(self._rand_idx)
        else:
            #the max_to_present secures that random examples
            #are returned from the same pool each time (in case
            #the total num of examples was limited by max_num_batches)
            max_to_present = self.batch_size*self._max_num_batches \
                                if self._max_num_batches > 0 else self.x.shape[0]
            return self.rng.permutation(numpy.arange(0, self.x.shape[0]))[0:max_to_present]

    def next(self):

        has_enough = (self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches <= (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        rval_x = self.x[range_idx]
        rval_t = self.t[range_idx]

        self._curr_idx += self.batch_size

        if self.conv_reshape:
            rval_x = rval_x.reshape(self.batch_size, 1,-1)

        return rval_x, self.__to_one_of_k(rval_t)

    def num_examples(self):
        return self.x.shape[0]

    def num_examples_presented(self):
        return self._curr_idx + 1

    def __to_one_of_k(self, y):
        rval = numpy.zeros((y.shape[0], self.num_classes), dtype=numpy.float32)
        #print rval, y, self.num_classes
        for i in xrange(y.shape[0]):
            rval[i, y[i]] = 1
        return rval

class MetOfficeDataProvider(DataProvider):
    """
    The class iterates over South Scotland Weather, in possibly
    random order.
    """
    def __init__(self, window_size,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True):

        super(MetOfficeDataProvider, self).\
            __init__(batch_size, randomize)

        dset_path = './data/HadSSP_daily_qc.txt'
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        if max_num_batches > 0 and max_num_examples > 0:
            logger.warning("You have specified both 'max_num_batches' and " \
                  "a deprecead 'max_num_examples' arguments. We will " \
                  "use the former over the latter.")

        raw = numpy.loadtxt(dset_path, skiprows=3, usecols=range(2, 32))

        self.window_size = window_size
        self._max_num_batches = max_num_batches
        #max_num_examples arg was provided for backward compatibility
        #but it maps us to the max_num_batches anyway
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size

        #filter out all missing datapoints and
        #flatten a matrix to a vector, so we will get
        #a time preserving representation of measurments
        #with self.x[0] being the first day and self.x[-1] the last
        self.x = raw[raw >= 0].flatten()

        #normalise data to zero mean, unit variance
        mean = numpy.mean(self.x)
        var = numpy.var(self.x)
        assert var >= 0.01, (
            "Variance too small %f " % var
        )
        self.x = (self.x-mean)/var

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MetOfficeDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        # we generate random indexes starting from window_size, i.e. 10th absolute element
        # in the self.x vector, as we later during mini-batch preparation slice
        # the self.x container backwards, i.e. given we want to get a training
        # data-point for 11th day, we look at 10 preeceding days.
        # Note, we cannot do this, for example, for the 5th day as
        # we do not have enough observations to make an input (10 days) to the model
        return numpy.random.permutation(numpy.arange(self.window_size, self.x.shape[0]))

    def next(self):

        has_enough = (self.window_size + self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches <= (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self.window_size + self._curr_idx,
                             self.window_size + self._curr_idx + self.batch_size)

        #build slicing matrix of size minibatch, which will contain batch_size
        #rows, each keeping indexes that selects windows_size+1 [for (x,t)] elements
        #from data vector (self.x) that itself stays always sorted w.r.t time
        range_slices = numpy.zeros((self.batch_size, self.window_size + 1), dtype=numpy.int32)

        for i in xrange(0, self.batch_size):
            range_slices[i, :] = \
                numpy.arange(range_idx[i],
                             range_idx[i] - self.window_size - 1,
                             -1,
                             dtype=numpy.int32)[::-1]

        #here we use advanced indexing to select slices from observation vector
        #last column of rval_x makes our targets t (as we splice window_size + 1
        tmp_x = self.x[range_slices]
        rval_x = tmp_x[:,:-1]
        rval_t = tmp_x[:,-1].reshape(self.batch_size, -1)

        self._curr_idx += self.batch_size

        return rval_x, rval_t


class FuncDataProvider(DataProvider):
    """
    Function gets as an argument a list of functions defining the means
    of a normal distribution to sample from.
    """
    def __init__(self,
                 fn_list=[lambda x: x ** 2, lambda x: numpy.sin(x)],
                 std_list=[0.1, 0.1],
                 x_from = 0.0,
                 x_to = 1.0,
                 points_per_fn=200,
                 batch_size=10,
                 randomize=True):
        """
        """

        super(FuncDataProvider, self).__init__(batch_size, randomize)

        def sample_points(y, std):
            ys = numpy.zeros_like(y)
            for i in xrange(y.shape[0]):
                ys[i] = numpy.random.normal(y[i], std)
            return ys

        x = numpy.linspace(x_from, x_to, points_per_fn, dtype=numpy.float32)
        means = [fn(x) for fn in fn_list]
        y = [sample_points(mean, std) for mean, std in zip(means, std_list)]

        self.x_orig = x
        self.y_class = y

        self.x = numpy.concatenate([x for ys in y])
        self.y = numpy.concatenate([ys for ys in y])

        if self.randomize:
            self._rand_idx = self.__randomize()
        else:
            self._rand_idx = None

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        return numpy.random.permutation(numpy.arange(0, self.x.shape[0]))

    def __iter__(self):
        return self

    def next(self):
        if (self._curr_idx + self.batch_size) >= self.x.shape[0]:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        x = self.x[range_idx]
        y = self.y[range_idx]

        self._curr_idx += self.batch_size

        return x, y


class MNISTDataProvider(DataProvider):
    """
    The class iterates over MNIST digits dataset, in possibly
    random order.
    """
    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 max_num_examples=-1,
                 randomize=True,
                 rng=None,
                 conv_reshape=False):

        super(MNISTDataProvider, self).\
            __init__(batch_size, randomize, rng)

        assert dset in ['train', 'valid', 'eval'], (
            "Expected dset to be either 'train', "
            "'valid' or 'eval' got %s" % dset
        )
        
        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )
        
        if max_num_batches > 0 and max_num_examples > 0:
            logger.warning("You have specified both 'max_num_batches' and " \
                  "a deprecead 'max_num_examples' arguments. We will " \
                  "use the former over the latter.")

        dset_path = './data/mnist_%s.pkl.gz' % dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        with gzip.open(dset_path) as f:
            x, t = cPickle.load(f)

        self._max_num_batches = max_num_batches
        #max_num_examples arg was provided for backward compatibility
        #but it maps us to the max_num_batches anyway
        if max_num_examples > 0 and max_num_batches < 0:
            self._max_num_batches = max_num_examples / self.batch_size      
            
        self.x = x
        self.t = t
        self.num_classes = 10
        self.conv_reshape = conv_reshape

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MNISTDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)

        if self._rand_idx is not None and self._max_num_batches > 0:
            return self.rng.permutation(self._rand_idx)
        else:
            #the max_to_present secures that random examples
            #are returned from the same pool each time (in case
            #the total num of examples was limited by max_num_batches)
            max_to_present = self.batch_size*self._max_num_batches \
                                if self._max_num_batches > 0 else self.x.shape[0]
            return self.rng.permutation(numpy.arange(0, self.x.shape[0]))[0:max_to_present]

    def next(self):

        has_enough = (self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches <= (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        rval_x = self.x[range_idx]
        rval_t = self.t[range_idx]

        self._curr_idx += self.batch_size

        if self.conv_reshape:
            rval_x = rval_x.reshape(self.batch_size, 1, 28, 28)

        return rval_x, self.__to_one_of_k(rval_t)

    def num_examples(self):
        return self.x.shape[0]

    def num_examples_presented(self):
        return self._curr_idx + 1

    def __to_one_of_k(self, y):
        rval = numpy.zeros((y.shape[0], self.num_classes), dtype=numpy.float32)
        for i in xrange(y.shape[0]):
            rval[i, y[i]] = 1
        return rval
