import numpy as np
import scipy.optimize as sci_opt
import scipy.stats as sci_stats
import matplotlib.pyplot as plt

# Much of the following is inspired by "Statically Transformed Autoregressive
# Process and Surrogate Data Test for Nonlinearity", D. Kugiumtzis (2002)

class MarginalTransform(object):
	"""MarginalTransform generates a polynomial transform from Gaussian to provided random variable.

	"""

	def __init__(self, dim = None):
		"""Initializes the MarginalTransform.

		"""
		self._poly = None
		self._dim = dim

	def fit(self, samples, diagnostics = False):
		"""Fits the polynomial coefficients of a marginal transform from samples.

		"""
		# 0. Compute an empirical CDF of the provided samples
		vals = np.sort(samples)
		cdf = np.arange(len(vals))/float(len(vals))

		# 1. Draw Ui, Zi samples via uniform and inverse CDF
		N = 10000
		r = np.random.rand(N, 1).reshape((N,))
		# TODO: Maybe use importance sampling to sample places where CDF of Z and/or U are flatter
		u = sci_stats.norm.ppf(r)
		z = 0.*r
		for ind, ri in enumerate(r):
			cdf_ind = (np.abs(cdf - ri)).argmin()
			z[ind] = vals[cdf_ind]

		# 2. Fit a polynomial to the data (Ui, Zi)
		# TODO: This could be improved by using Chebyshev or Legendry polynomials
		# and then pulling out the coefficients by monomial
		if self._dim is None:
			# Cross validate to find dim hyperparameter
			# by minimizing the max oos MSE
			max_dims = 20
			k = 20
			train_perc = 0.80
			dims = range(1, max_dims)
			max_mses = [ 0. ] * (max_dims - 1)
			for ind, dim in enumerate(dims):
				for it in range(k):
					perm = np.random.permutation(N)
					train_inds = perm[:np.floor(train_perc*N).astype(int)]
					test_inds = perm[np.floor(train_perc*N).astype(int):]
					u_train = u[train_inds]
					u_test = u[test_inds]
					z_train = z[train_inds]
					z_test = z[test_inds]
					poly_coeff = np.polyfit(u_train, z_train, dim)
					poly = np.poly1d(poly_coeff)
					residuals = z_test - poly(u_test)
					mse = np.sum(np.square(residuals))
					if mse > max_mses[ind]:
						max_mses[ind] = mse
			self._dim = dims[np.asarray(max_mses).argmin()]
		
		self._a = np.polyfit(u, z, self._dim)	
		self._poly = np.poly1d(self._a)

		# 3. Check
		if diagnostics:
			sorted_u = np.sort(u)
			k = np.linspace(-3, 3, 1000)
			gk = self._poly(k)
			fig = plt.figure(1)
			fig.suptitle('Polynomial fit with d=%d' %(self._dim))
			plt.plot(u, z, '*b')
			plt.plot(k, gk, '-r')
			plt.grid()
			plt.xlabel('U(r)')
			plt.ylabel('Z(r)')
			
			plt.show()

	def sample(self, m=1):
		# Draw m normals
		u = np.random.randn(m,1)
		# Map through polynomial transform
		z = self._poly(u)
		return z

	def generate(self, m=1):
		assert self._poly is not None, "MarginalTransform must be trained first"
		yield self.sample(m)

	def __call__(self, u):
		return self._poly(u)


class AcfTransformError(Exception):
	pass

class AcfTransform(object):
	"""AcfTransform determines the ACF of a white noise process corresponding to provided ACF

	"""

	def __init__(self):
		self._cj = None

	def fit(self, a):
		# a is the vector of polynomial coefficients of the marginal transform
		# s.t.  Z = a_0 + sum_i a_i U_i^j
		m = len(a) - 1
		# Set up the coefficients c_j for p_X = sum_j c_j p_U^j
		
		# Central moments
		# mu_{2k+1} = 0, mu_{2k} = 1 * 3 * ... * (2k-1) for k>=0
		mu = np.zeros((2*m + 1,))
		for k in range(0, m + 1):
			mu[2*k] = np.prod(np.arange(1, 2*k, 2))
		
		# s-th, t-th joint moments (as vector of coefficients of monomials p_U^j)
		must = np.zeros((m + 1, m + 1, m + 1))
		max_k = np.floor(m/2.0).astype(int)
		for k in range(0, max_k + 1):
			for l in range(0, max_k + 1):
				if (k+l) % 2 == 1:
					continue
				for j in range(0, min(k, l) + 1):
					must[2*k, 2*l, 2*j] = np.math.factorial(2*k)*np.math.factorial(2*l)/np.power(2, k+l-2*j)/np.math.factorial(k-j)/np.math.factorial(l-j)/np.math.factorial(2*j)
					if 2*k + 1 <= m and 2*l + 1 <= m:
						must[2*k+1, 2*l+1, 2*j+1] = np.math.factorial(2*k + 1)*np.math.factorial(2*l + 1)/np.power(2, k+l-2*j)/np.math.factorial(k-j)/np.math.factorial(l-j)/np.math.factorial(2*j + 1)

		num = np.zeros((m,))
		den = 0.0
		for s in range(1, m+1):
			for t in range(1, m+1):
				den += a[s]*a[t]*(mu[s + t] - mu[s]*mu[t])
				num += a[s]*a[t]*(must[s,t][1:m+1] - mu[s]*mu[t])
		self._cj = num / den

	def __call__(self, acf):
		assert abs(acf[0] - 1.0) < 1e-10, "First element of ACF should be 1.0"
		# Maps provided ACF of sample series into corresponding white noise process ACF
		acf = np.asarray(acf)
		acf = acf[1:] # discard acf(0) = 1
		n = acf.shape[0]
		m = self._cj.shape[0]
		acf = acf.reshape((n,))
		def F(x):
			A = np.vander(x, m + 1, increasing=True)[:, 1:m+1] # discard unit column
			Ac = np.dot(A, self._cj)
			return acf - Ac

		x0 = np.random.randn(n).reshape((n,))
		normal_acf = sci_opt.broyden1(F, x0, f_tol = 1e-8, maxiter = 100)
		normal_acf = normal_acf.reshape((n, 1))
		return np.vstack((np.ones((1,1)), normal_acf))


class AutoregressiveModel(object):
	"""AutoregressiveModel generates a white-noise AR series with provided ACF.
	
	"""

	def __init__(self):
		"""Initializes the AutoregressiveModel.

		"""
		self._ar_coeff = None
		self._ar_memory = None

	def fit(self, acf):
		"""Fits the AR coefficients for provided ACF.

		"""
		# Solve for coefficients and noise std
		# We assume the best model order is p=n for ACF of length n
		assert abs(acf[0] - 1.0) < 1e-10, "First element of ACF should be 1.0"
		n = acf.shape[0]
		A = np.zeros((n - 1, n - 1))
		for k in range(-n + 2, n - 1):
			v = np.asscalar(acf[abs(k)]) * np.ones((n - 1 - abs(k),))
			Ad = np.diag(v, k)
			A += Ad 
		b = np.asarray(acf[1:]).reshape((n - 1,))
		self._ar_coeff = np.linalg.solve(A, b)
		self._noise_std = np.sqrt(1.0 - np.dot(self._ar_coeff, acf[1:]))

	def next(self):
		next_sample = np.dot(self._ar_coeff, self._ar_memory) + self._noise_std * np.random.randn()
		self._ar_memory = np.vstack((next_sample, self._ar_memory[:-1]))
		return next_sample

	def sample(self, m=1):
		assert self._ar_coeff is not None, "The model must be trained before samples can be generated"
	
		n = len(self._ar_coeff)
		if self._ar_memory is None: # Burn in the series
			self._ar_memory = np.random.randn(n, 1)
			for j in range(10*n):
				self.next()
	
		samples = np.empty((m,))
		for k in range(m):
			samples[k] = self.next()
		
		if m == 1:
			samples = np.asscalar(samples)
		return samples

	def generate(self, m=1):
		yield self.sample(m)

class TimeSeriesGenerator(object):
	"""TimeSeriesGenerator simulates series with statistics of provided series.

	"""

	def __init__(self):
		"""Initializes the TimeSeriesGenerator.

		"""
		self._marginal_transform = None
		self._acf_trans_coeff = None
		self._ar_model = None
		self._is_trained = False

	def fit(self, series, max_lag = 4):
		"""Fits a TimeSeriesGenerator to the marginal CDF and 
		autocorrelation function of the provided series.

		The provided series should be wide-sense stationary

		Args:
			series: N-dimensional numpy array

		Returns:
			(None)

		"""
		def autocorr(x, t=1):
			x = x.reshape(1, len(x))
			p1 = x[0, :max(x.shape)-t]
			p2 = x[0, t:]
			return np.corrcoef(p1, p2)[0, 1]
		
		# 0. Find the acf of the provided series
		acf = np.asarray([autocorr(series, t) for t in range(max_lag)])

		K = 30
		marginal_transforms = [None] * K
		acf_transforms = [None] * K
		ar_models = [None] * K
		normal_acfs = [None] * K
		mse_acfs = np.zeros((K,))

		for l in range(K):
			print("Sampling %d of %d iterations" %(l, K))
			try:
				# 1. Determine the marginal transform g s.t. Z = g(U)
				marginal_transforms[l] = MarginalTransform()
				marginal_transforms[l].fit(series)
			
				# 2. Find the coefficients mapping normal ACF to series ACF
				a = marginal_transforms[l]._a[::-1]
				acf_transforms[l] = AcfTransform()
				acf_transforms[l].fit(a)
				normal_acfs[l] = acf_transforms[l](acf)

				# 3. Construct an AR(p) model with the normal ACF statistics
				ar_models[l] = AutoregressiveModel()
				ar_models[l].fit(normal_acfs[l])

				# 4. Generate samples from the AR model and map through g
				test_samples = marginal_transforms[l](ar_models[l].sample(10000))
				test_acf = np.asarray([autocorr(test_samples, t) for t in range(max_lag)])

				mse_acfs[l] = np.sum(np.square(acf-test_acf))
				if np.isnan(mse_acfs[l]):
					mse_acfs[l] = float("inf")

			except sci_opt.nonlin.NoConvergence as e:
				print(e)
				mse_acfs[l] = float("inf")

		best_l = mse_acfs.argmin()

		# 6. Store the marginal transform and AR(p) coefficients
		self._marginal_transform = marginal_transforms[best_l]
		self._acf_transform = acf_transforms[best_l]
		self._ar_model = ar_models[best_l]
		self._is_trained = True

		# Log
		print("========SUMMARY=======")
		print("  marginal transform coefficients =", self._marginal_transform._a)
		print("  ar model coefficients =", self._ar_model._ar_coeff)
		print("  normal_acf =", normal_acfs[best_l])
		print("----------------------")

	def sample(self, m=1):
		assert self._is_trained, "The generator has not been trained"
		ar_samples = self._ar_model.sample(m)
		transformed_samples = self._marginal_transform(ar_samples)
		return transformed_samples

	def generate(self, m=1):
		"""Generator of the next m samples in the simulated time series.

		Args:
			(None)

		Yields:
			samples (m-dimensional numpy array): The next sample m samples in the series

		"""
		assert self._is_trained, "The generator has not been trained"
		# Yield next m samples in the series
		yield self.sample(m)


if __name__ == '__main__':
	import sys

	TEST_CASE = sys.argv[1]

	max_lag = 10
	burn = 10*max_lag
	Ns = 10000

	if TEST_CASE == "armodel":
		print("Running armodel test case...")
		train_samples = np.zeros((Ns + burn, 1))
		train_samples[:max_lag,0] = np.random.randn(max_lag)
		for i in range(max_lag, Ns+burn):
			train_samples[i] = 1.5 - 0.5*train_samples[i-1] + np.random.randn()
		train_samples = train_samples[burn:].reshape((Ns,))

	elif TEST_CASE == "uniform":
		print("Running uniform test case...")
		train_samples = 0.4 * np.random.rand(Ns) + 3.

	else:
		print("ERROR: Unrecognized test case '%s'" %(TEST_CASE))
		sys.exit(1)

	tsg = TimeSeriesGenerator()
	tsg.fit(train_samples, max_lag = max_lag)

	test_samples = tsg.sample(train_samples.shape[0])

	fig1 = plt.figure(1)
	fig1.suptitle('mu=%1.3f, std=%1.3f' %(np.mean(train_samples), np.std(train_samples)))
	plt.hist(train_samples, bins=100)
	plt.xlabel('train sample')
	plt.ylabel('freq')

	fig2 = plt.figure(2)
	fig2.suptitle('mu=%1.3f, std=%1.3f' %(np.mean(test_samples), np.std(test_samples)))
	plt.hist(test_samples, bins=100)
	plt.xlabel('test sample')
	plt.ylabel('freq')

	def autocorr(x, t=1):
		x = x.reshape(1, len(x))
		p1 = x[0, :max(x.shape)-t]
		p2 = x[0, t:]
		return np.corrcoef(p1, p2)[0, 1]

	fig3 = plt.figure(3)
	fig3.suptitle('Comparing train/test autocorrelations')
	train_acf = np.asarray([autocorr(train_samples, t) for t in range(max_lag)])
	test_acf = np.asarray([autocorr(test_samples, t) for t in range(max_lag)])
	plt.plot(train_acf, '*r')
	plt.plot(test_acf, 'ob')
	plt.plot(np.abs(train_acf-test_acf), 'dk')
	plt.legend(['train', 'test', 'error'])
	plt.grid()
	plt.xlabel('lag t')
	plt.ylabel('ACF(t)')

	plt.show()
