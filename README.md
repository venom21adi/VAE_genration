# VAE_genration
# Project Description

**Project Overview:**
This is a hobby project focused on building a Variational Autoencoder (VAE) model using TensorFlow's framework and Azure Machine Learning Studio. The project's primary goal is to generate new images from open-source image datasets. Please note that some images are currently incomplete due to data loss, but the code is being updated to produce new images.



**About Variational Autoencoder (VAE):**
A Variational Autoencoder (VAE) is a type of generative model used in machine learning and deep learning. It is a specific kind of autoencoder designed for unsupervised learning and dimensionality reduction. VAEs are particularly useful for generating new data similar to a given dataset. They find applications in tasks such as image generation, data denoising, and feature learning.

**Key Concepts of VAE:**
- **Encoder Network:** The encoder maps input data to mean (µ) and standard deviation (σ) vectors, which represent the parameters of a Gaussian distribution in the latent space.
- **Sampling:** A point is sampled from the latent space Gaussian distribution defined by µ and σ, introducing stochasticity.
- **Decoder Network:** The sampled point is passed through the decoder network to reconstruct the original input data.
- **Loss Function:** VAEs are trained using a loss function that includes a reconstruction loss and a regularization term. The regularization term, often the Kullback-Leibler (KL) divergence, encourages a structured latent space distribution.

**Important Note:**
Since this project was developed using Azure Machine Learning Studio, we have deliberately omitted the storage account details for security reasons.

**Repository Structure:**
- `Version 1`: Python script containing the VAE model implementation and two different configurations and their results. This is a structure where no batch normalization was implemented. One can see even how poorly it performs.
- `Version 2`: Python script containing the VAE model implementation and two different configurations and their results. This is a structure where batch normalization was implemented. One can see better it performs.

**Coming Soon:**
We are working on updating the project to produce new images, so stay tuned for exciting results.

**Sample Images:**
Here are some sample images from the previous project work:


<!-- Include sample image URLs here -->

---


