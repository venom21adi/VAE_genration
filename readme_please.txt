What is this project about?

This is a hobby project in which I am building a VAE model to use open source images as an input and create different images using Tensorflow's framework and Azure machine learning studio.


Why images are incomplete?

I had lost some of my data and trying to recreate the model again. So you'll see new images with updated code soon.
I have attached some images from my previous work as well.

About the code:
Since this was developed using Azure machine learning studio, I am deliberately skipping the storage account details.

What is variational auto encoder?

A Variational Autoencoder (VAE) is a type of generative model used in machine learning and deep learning. It is a specific kind of autoencoder, which is a neural network architecture designed for unsupervised learning and dimensionality reduction. VAEs are particularly useful for generating new data that is similar to a given dataset. They are often applied to tasks such as image generation, data denoising, and feature learning.

The key idea behind a Variational Autoencoder is to make the latent space of the model continuous and structured. In a traditional autoencoder, the encoder maps the input data to a fixed-size latent vector, which can be randomly distributed. In contrast, a VAE models the latent space as a probability distribution, typically a Gaussian distribution, which makes it possible to sample new data points from this distribution.

Here's how a VAE works:

1. Encoder Network: The encoder takes an input data point and maps it to two vectors: a mean vector (µ) and a standard deviation vector (σ). These vectors represent the parameters of a Gaussian distribution in the latent space.

2. Sampling: A point is sampled from the Gaussian distribution defined by µ and σ. This step introduces stochasticity, making the latent space continuous and structured.

3. Decoder Network: The sampled point is passed through the decoder network, which attempts to reconstruct the original input data.

4. Loss Function: VAEs are trained using a loss function that consists of two components: a reconstruction loss and a regularization term. The reconstruction loss measures how well the VAE can reconstruct the input data, while the regularization term encourages the latent space to be well-behaved and structured.

The regularization term is often the Kullback-Leibler (KL) divergence, which encourages the latent space distribution to be close to a standard Gaussian distribution. This helps in generating meaningful data points during sampling.

The beauty of VAEs is that they can be used for various generative tasks, such as generating new images that are similar to a given dataset, interpolating between data points, and more. They also have the advantage of being able to generate data points that do not exactly match any specific training example, making them useful for creative tasks like image generation and data synthesis.

