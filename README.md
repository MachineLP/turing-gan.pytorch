# Turing Generative Adversarial Network
[Turing GANs](https://arxiv.org/abs/1810.10948) are quick to train! This excited me to write my own versions in [PyTorch](https://pytorch.org) which is based on the original Keras [implementation](https://github.com/bojone/T-GANs).

## Experiments
So, following are my experiments' resulting image data.

### Note
- For all the experiments the images shown below are sampled after 100K iterations of training the Turing GAN on various datasets. 
- All the experiments used spectral normalization for 1-Lipschitz contraint enforcement. 
- I trained all of the Turing GANs with both Jensen-Shannon and Wasserstein divergences.

I performed experiments on the following datasets:
- CIFAR-10
- MNIST
- Fashion-MNIST

### CIFAR-10
#### Turing Standard GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/cifar-10/latest_100000.png)
#### Turing Wasserstein GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/cifar-10/latest_100000.png)

### MNIST
#### Turing Standard GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/mnist/latest_100000.png)
#### Turing Wasserstein GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/mnist/latest_100000.png)

### Fashion MNIST
#### Turing Standard GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/sgan/samples/fashion-mnist/latest_100000.png)
#### Turing Wasserstein GAN with Spectral Normalization
![](https://github.com/rahulbhalley/turing-gan.pytorch/raw/master/wgan/samples/fashion-mnist/latest_100000.png)

## References
- Training Generative Adversarial Networks Via Turing Test [[arXiv](https://arxiv.org/abs/1810.10948)]
- Original [T-GANs](https://github.com/bojone/T-GANs) implementation
- Spectral Normalization for Generative Adversarial Networks [[arXiv](https://arxiv.org/abs/1802.05957)]
- Spectral Normalization [implementation](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py) in [PyTorch](https://pytorch.org)
