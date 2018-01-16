


if __name__ == "__main__":
    from mlp.utils import *
    from mlp.conv import *
    from mlp.dataset import MNISTDataProvider
   

    rng = numpy.random.RandomState([2015,10,10])


    tl = ConvLinear_Opt(3,2, rng=rng, image_shape=(4, 4),kernel_shape=(2,2))

    test = test_conv_linear_fprop(tl)
    if test:
        print("Passed fprop")
    test = test_conv_linear_bprop(tl)
    if test:
        print("Passed bprop")
    test = test_conv_linear_pgrads(tl)
    if test:
        print("Passed pgrads")
