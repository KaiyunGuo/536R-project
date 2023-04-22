import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from torch.distributions import relaxed_categorical as rc
from torch.autograd import Variable
from utils.distributions import log_Normal_standard, log_Normal_diag
import math
import abc
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DSVAE(torch.nn.Module):
	def __init__(self, batch_size, input_dim, latent_dim, hidden_dim=300,
	       W_init=None, beta_vae=2, dropout_prob=0., row_normalized = False, **kwargs):
		super(DSVAE, self).__init__()

		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.dropout_prob = dropout_prob
		self.log_sigmas = nn.Parameter(torch.randn(input_dim))
		self.t_drop = nn.Dropout(dropout_prob)
		self.beta_vae = beta_vae
		self.row_normalized = row_normalized

		if W_init is not None:
			self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float, device=device))
		else:
			self.W = nn.Parameter(torch.randn(input_dim, latent_dim), requires_grad=True)

		# Encoder layers
		self.q_z = nn.Sequential(nn.Linear(input_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU(),
								 nn.Linear(hidden_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU())

		self.z_mean = nn.Linear(hidden_dim, latent_dim)
		self.z_log_var = nn.Linear(hidden_dim, latent_dim)

		# Decoder layers
		self.z_generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
										nn.ReLU(),
										nn.Linear(hidden_dim, hidden_dim),
										nn.ReLU())
		self.m_generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
				   						nn.ReLU())

		### spike-and-slab prior
		# hyperparameters that control the balance between sparse (ψ₀) and dense (ψ₁) weight elements.
		self.lambda0 = kwargs.get('lambda0', 10)
		self.lambda1 = kwargs.get('lambda1', 0.1)

		self.a = kwargs.get('a', 1.)
		self.b = kwargs.get('b', self.input_dim)

		# the probability that each weight comes from the slab part of the spike-and-slab prior
		self.p_star = nn.Parameter(0.5 * torch.ones(self.input_dim, self.latent_dim,
					      			dtype=torch.float, device=device), requires_grad=False)
		
		# mixture probabilities for the spike component of the spike-and-slab prior in each latent dimension
		# entry corresponds to the overall probability that a weight in latent dimension belongs to the spike component
		self.thetas = nn.Parameter(torch.rand(self.latent_dim), requires_grad=False)


	def encode(self, x):
		q_z = self.q_z(x)
		z_mean = self.z_mean(q_z)
		z_log_var = self.z_log_var(q_z)
		return z_mean, z_log_var

	def reparameterize(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		sample = mean + (eps * std)
		return sample

	def decode(self, z):
		mask = self.get_generator_mask() # mask: (G, K)

		x_reconstructed = self.z_generator(z)
		masked = self.m_generator(mask)
		x_reconstructed = torch.mm(x_reconstructed, torch.transpose(masked, 0, 1))
		
		return x_reconstructed
	
	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		z = self.reparameterize(z_mean, z_log_var)
		x_mean = self.decode(z)

		return x_mean, z, z_mean, z_log_var

	def kld_z(self, z, mu, log_var):
		kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
		
		kld = self.beta_vae * kld
		return kld

	# the prior
	def log_p_z(self, z):
		### used when z prior is mixed Gaussian
		log_prior = log_Normal_standard(z, dim=1)
		return log_prior

	# reconstruction loss
	def reconstruction_loss(self, x, recon_x):
		# loss_rec = F.mse_loss(recon_x, x, reduction='sum')
		# loss_rec = -torch.sum(
        #     (-0.5 * np.log(2.0 * np.pi))
        #     + (-0.5 * logvar_x)
        #     + ((-0.5 / torch.exp(logvar_x)) * (x - recon_x) ** 2.0)
        #     )
	
		sigmas = torch.exp(self.log_sigmas)
		loss = nn.MSELoss()
		loss_rec = 0.5 * loss(recon_x / sigmas, (x / sigmas))

		return loss_rec

	def vae_loss(self, x):
		### Train function
		x_mean, z, z_mean, z_log_var = self.forward(x)

		kl_loss = self.kld_z(z, z_mean, z_log_var)
		w_loss = self.mask_loss()
		reconstruction_loss = self.reconstruction_loss(x, x_mean)

		return reconstruction_loss, kl_loss, w_loss

	def get_generator_mask(self):
		if self.row_normalized:
			# re-scale so rows sum to 1
			W = self.W.abs() + 1e-6
			W = F.normalize(self.W.abs(), p=1, dim=-1)
		else:
			W = self.W
		return W

	def mask_loss(self):
		loss = (self.lambda1 * self.p_star + self.lambda0 * (1 - self.p_star)) * self.W.abs()
		loss = loss.sum()

		loss = loss / self.batch_size
		return loss


class BaseSparseVAE(torch.nn.Module):
	def __init__(self, batch_size, input_dim, latent_dim, hidden_dim=300,
				 z_prior='standard', W_init=None, sigma_prior_scale=1., sigma_prior_df=3,
				 loss_type='mse', sigmas_init=None, dropout_prob=0., n_pseudo=None, row_normalize=True, **kwargs):
		super(BaseSparseVAE, self).__init__()

		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.dropout_prob = dropout_prob		# not used?
		self.t_drop = nn.Dropout(dropout_prob)	# not used?
		self.sigma_prior_df = sigma_prior_df
		self.sigma_prior_scale = sigma_prior_scale
		self.loss_type = loss_type
		self.row_normalize = row_normalize
		self.z_prior = z_prior		# 'standard', 'covariance', 'vampprior'

		if W_init is not None:
			self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float, device=device))
		else:
			self.W = nn.Parameter(torch.randn(input_dim, latent_dim), requires_grad=True)

		if n_pseudo is None:
			self.n_pseudo = 50  # vampprior default is 500.
		else:
			self.n_pseudo=n_pseudo

		if sigmas_init is not None:
			self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas_init, dtype=torch.float)))
		else:
			self.log_sigmas = nn.Parameter(torch.randn(input_dim))

		# Encoder layers
		self.q_z = nn.Sequential(nn.Linear(input_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU(),
								 nn.Linear(hidden_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU())

		self.z_mean = nn.Linear(hidden_dim, latent_dim)
		self.z_log_var = nn.Linear(hidden_dim, latent_dim)

		# Decoder layers
		self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim, bias=False),
										# nn.BatchNorm1d(hidden_dim, affine=False),
										nn.ReLU(),
										nn.Linear(hidden_dim, hidden_dim),
										nn.ReLU())
		# produce the mean of each dimension of the reconstructed x separately
		### why???
		self.column_means = nn.ModuleList([nn.Linear(hidden_dim, 1) for i in range(input_dim)])


	@abstractmethod
	def get_generator_mask(self):
		pass

	def encode(self, x):
		q_z = self.q_z(x)
		z_mean = self.z_mean(q_z)
		z_log_var = self.z_log_var(q_z)
		return z_mean, z_log_var

	def reparameterize(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		# error: standard normal dist
		eps = torch.randn_like(std)
		sample = mean + (eps * std)
		return sample

	def decode(self, z):
		x_reconstructed = torch.zeros(z.shape[0], self.input_dim, device=device)
		mask = self.get_generator_mask()
		# print("decoder mask.shape: ", mask.shape)

		for j in range(self.input_dim):
			# apply different masks and linear layers from self.column_means
			# to the output of the generator network for each input feature
			masked_input = torch.mul(z, mask[j, :])
			x_reconstructed[:,j] = self.column_means[j](self.generator(masked_input)).squeeze()

		del masked_input
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	
		return x_reconstructed

	def kld_z(self, z, mu, log_var):

		if self.z_prior == 'standard':
			kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

		else:
			raise Exception('Wrong name of the prior!')

		return kld

	# the prior
	def log_p_z(self, z):
		### used when z prior is mixed Gaussian
		if self.z_prior == 'standard':
			log_prior = log_Normal_standard(z, dim=1)

		return log_prior

	# reconstruction loss
	def reconstruction_loss(self, x_pred, x):

		# loss_rec = -torch.sum(
        #     (-0.5 * np.log(2.0 * np.pi))
        #     + (-0.5 * logvar_x)
        #     + ((-0.5 / torch.exp(logvar_x)) * (x - recon_x) ** 2.0)
        #     )

		if self.loss_type == 'mse':
			# Using this!!!
			sigmas = torch.exp(self.log_sigmas)
			loss = nn.MSELoss()
			reconstruction_loss = 0.5 * loss(x_pred / sigmas, (x / sigmas))

		return reconstruction_loss


	@abstractmethod
	def mask_loss(self):
		pass

	def sigma_loss(self):
		# sig_loss = (self.batch_size + self.sigma_prior_df + 2) * self.log_sigmas.sum() \
		# 	+ 0.5 * self.sigma_prior_df * self.sigma_prior_scale * torch.sum(1/torch.exp(2 * self.log_sigmas))

		# sig_loss = sig_loss / self.batch_size
		# return sig_loss
		pass

	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		z = self.reparameterize(z_mean, z_log_var)
		x_mean = self.decode(z)

		return x_mean, z, z_mean, z_log_var

	def vae_loss(self, x, normalized_x=None):
		### Train function
		if normalized_x is not None:
			x_mean, z, z_mean, z_log_var = self.forward(normalized_x)
		else:
			x_mean, z, z_mean, z_log_var = self.forward(x)

		kl_loss = self.kld_z(z, z_mean, z_log_var)
		w_loss = self.mask_loss()
		reconstruction_loss = self.reconstruction_loss(x_mean, x)

		if self.loss_type == 'mse':
			sigma_loss = self.sigma_loss()
		else:
			sigma_loss = torch.tensor([0.], dtype=torch.float, device=device)

		return reconstruction_loss, kl_loss, w_loss, sigma_loss


class SparseVAESpikeSlab(BaseSparseVAE):
	def __init__(self, *args, **kwargs):
		super(SparseVAESpikeSlab, self).__init__(*args, **kwargs)

		# hyperparameters that control the balance between sparse (ψ₀) and dense (ψ₁) weight elements.
		self.lambda0 = kwargs.get('lambda0', 10)
		self.lambda1 = kwargs.get('lambda1', 0.1)


		self.a = kwargs.get('a', 1.)
		self.b = kwargs.get('b', self.input_dim)

		# the probability that each weight comes from the slab part of the spike-and-slab prior
		self.p_star = nn.Parameter(0.5 * torch.ones(self.input_dim, self.latent_dim,
					      			dtype=torch.float, device=device), requires_grad=False)
		
		# mixture probabilities for the spike component of the spike-and-slab prior in each latent dimension
		# entry corresponds to the overall probability that a weight in latent dimension belongs to the spike component
		self.thetas = nn.Parameter(torch.rand(self.latent_dim), requires_grad=False)

	def get_generator_mask(self):
		if self.row_normalize:
			# re-scale so rows sum to 1
			W = self.W.abs() + 1e-6
			W = F.normalize(self.W.abs(), p=1, dim=-1)
		else:
			W = self.W
		return W

	def mask_loss(self):
		loss = (self.lambda1 * self.p_star + self.lambda0 * (1 - self.p_star)) * self.W.abs()
		loss = loss.sum()

		loss = loss / self.batch_size
		return loss


class VAE(torch.nn.Module):
	def __init__(self, batch_size, input_dim, latent_dim, hidden_dim=300,
				 z_prior='standard', W_init=None, sigma_prior_scale=1., sigma_prior_df=3,
				 loss_type='mse', sigmas_init=None, dropout_prob=0., n_pseudo=None, beta_vae=1, add_skip=False,**kwargs):
		super(VAE, self).__init__()

		self.batch_size = batch_size
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.dropout_prob = dropout_prob
		self.t_drop = nn.Dropout(dropout_prob)
		# self.sigma_prior_df = sigma_prior_df
		# self.sigma_prior_scale = sigma_prior_scale
		self.loss_type = loss_type
		self.beta_vae=beta_vae
		self.add_skip=add_skip

		self.W = torch.ones(self.input_dim, self.latent_dim, device=device)

		self.z_prior = z_prior
		if n_pseudo is None:
			self.n_pseudo = 50  # vampprior default is 500...
		else:
			self.n_pseudo=n_pseudo

		if sigmas_init is not None:
			self.log_sigmas = nn.Parameter(torch.log(torch.tensor(sigmas_init, dtype=torch.float)))
		else:
			self.log_sigmas = nn.Parameter(torch.randn(input_dim))

		self.q_z = nn.Sequential(nn.Linear(input_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU(),
								 nn.Linear(hidden_dim, hidden_dim),
								 # nn.BatchNorm1d(hidden_dim),
								 nn.ReLU())

		self.z_mean = nn.Linear(hidden_dim, latent_dim)
		self.z_log_var = nn.Linear(hidden_dim, latent_dim)

		self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim, bias=False),
										# nn.BatchNorm1d(hidden_dim, affine=False),
										nn.ReLU(),
										nn.Linear(hidden_dim, hidden_dim),
										nn.ReLU(),
									    nn.Linear(hidden_dim, input_dim))

		if add_skip==True:
			self.skip_layer = nn.Linear(latent_dim, input_dim)

		if z_prior == 'covariance':
			self.C = nn.Parameter(torch.randn(self.latent_dim, 2))

		# pseudo-inputs for vamp prior
		if z_prior == 'vampprior':

			# instead of modeling pseudo-inputs directly, model f(I) = u, where I is identity, f is learned and u are psuedo
			# this helps if domain of x is restricted
			if loss_type=="mse":
				self.vp_means = nn.Linear(self.n_pseudo, self.input_dim, bias=False)

			if loss_type=="binary":
				self.vp_means = nn.Sequential(nn.Linear(self.n_pseudo, self.input_dim, bias=False),
											  nn.Sigmoid())

			if loss_type=='categorical':
				self.vp_means = nn.Sequential(nn.Linear(self.n_pseudo, self.input_dim, bias=False),
											  nn.Softmax())

			# create an idle input for calling pseudo-inputs
			self.idle_input = Variable(torch.eye(self.n_pseudo, self.n_pseudo, requires_grad=False))


	def encode(self, x):
		q_z = self.q_z(x)
		z_mean = self.z_mean(q_z)
		z_log_var = self.z_log_var(q_z)
		return z_mean, z_log_var

	def reparameterize(self, mean, log_var):
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		sample = mean + (eps * std)
		return sample

	def decode(self, z):
		if self.add_skip:
			x_reconstructed = self.generator(z) + self.skip_layer(z)
		else:
			x_reconstructed = self.generator(z)

		if self.loss_type=='binary':
			x_reconstructed = torch.sigmoid(x_reconstructed)

		if self.loss_type == 'categorical':
			x_reconstructed = F.softmax(x_reconstructed, dim=-1)

		return x_reconstructed

	def kld_z(self, z, mu, log_var):

		if self.z_prior == 'standard':
			kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

		elif self.z_prior == 'vampprior':
			log_p_z = self.log_p_z(z)
			log_q_z = log_Normal_diag(z, mu, log_var, dim=1)
			kld = -(log_p_z - log_q_z)
			kld = torch.mean(kld)

		elif self.z_prior == 'covariance':

			z_cov = torch.matmul(self.C, torch.transpose(self.C, 0, 1)) + torch.eye(self.latent_dim)
			z_cov_inv = torch.inverse(z_cov)

			z_cov_diag = torch.diagonal(z_cov_inv)
			kld = (z_cov_diag * log_var.exp()).sum()

			for i in range(z.shape[0]):
				kld += torch.matmul(torch.matmul(mu[i, :], z_cov_inv), mu[i, :])

			kld += self.batch_size * torch.log(torch.det(z_cov))
			kld -= torch.sum(log_var)

			kld = 0.5 * kld / self.batch_size

		else:
			raise Exception('Wrong name of the prior!')
		
		kld = self.beta_vae * kld
		return kld

	# the prior
	def log_p_z(self, z):
		if self.z_prior == 'standard':
			log_prior = log_Normal_standard(z, dim=1)

		elif self.z_prior == 'vampprior':
			# z - MB x M

			# calculate psuedo inputs
			u = self.vp_means(self.idle_input)

			# calculate params for given data
			z_p_mean, z_p_logvar = self.encode(u)

			# expand z
			z_expand = z.unsqueeze(1)
			means = z_p_mean.unsqueeze(0)
			logvars = z_p_logvar.unsqueeze(0)

			a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(self.n_pseudo)  # MB x C
			a_max, _ = torch.max(a, 1)  # MB x 1

			# calculte log-sum-exp
			log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))

		else:
			raise Exception('Wrong name of the prior!')

		return log_prior

	# reconstruction loss
	def reconstruction_loss(self, x_pred, x):
		if self.loss_type == 'mse':
			sigmas = torch.exp(self.log_sigmas)
			loss = nn.MSELoss()
			reconstruction_loss = 0.5 * loss(x_pred / sigmas, (x / sigmas))

		return reconstruction_loss

	def sigma_loss(self):
		# sig_loss = (self.batch_size + self.sigma_prior_df + 2) * self.log_sigmas.sum() \
		# 	+ 0.5 * self.sigma_prior_df * self.sigma_prior_scale * torch.sum(1/torch.exp(2 * self.log_sigmas))

		# sig_loss = sig_loss / self.batch_size
		# return sig_loss
		pass

	def forward(self, x):
		z_mean, z_log_var = self.encode(x)
		z = self.reparameterize(z_mean, z_log_var)
		x_mean = self.decode(z)

		return x_mean, z, z_mean, z_log_var

	def vae_loss(self, x, normalized_x=None):
		if normalized_x is not None:
			x_mean, z, z_mean, z_log_var = self.forward(normalized_x)
		else:
			x_mean, z, z_mean, z_log_var = self.forward(x)

		kl_loss = self.kld_z(z, z_mean, z_log_var)
		reconstruction_loss = self.reconstruction_loss(x_mean, x)

		if self.loss_type == 'mse':
			sigma_loss = self.sigma_loss()
		else:
			sigma_loss = torch.tensor([0.], dtype=torch.float, device=device)

		w_loss = torch.tensor([0.], dtype=torch.float, device=device)

		return reconstruction_loss, kl_loss, w_loss, sigma_loss

	def get_generator_mask(self):

		return self.W
