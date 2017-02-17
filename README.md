# MLPHonoursExtension

This repository implements an adaptive Discrete Fourier Spectrum via incorporating the transformation as a layer in the neural network f: C -> R .

The gradients modify the complex functions complex weight space
by establishing an isomorphism \phi : C -> R^2 . Furthermore while
the spectrums domain is in the complex plane its codomain is just a
real number (set up partially motivated by Liouville's theorem) thus we have no unordered set problems as if we were to use a complex codomain. 

At a high level we have a layer that is preset to compute a fourier spectrum and is adapted via the backpropagation of the neural network. This is introducing the idea of a spectral prior taking a partial motivation in pretraining and regularisation but instead we preset in accordance to our beliefes which are that the input is periodic. 

The mathematics can be found in fourier_analysis_machine.pdf and the code in mlp_source_code . 

Some of the code was modified from the skeleton provided by Edinburgh University Machine Learning Practical (INFR11119) @PawelSwietojanski
