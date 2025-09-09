import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def rbf_kernel(x1, x2, length_scale=1.0, stdev=1.0):
    dx = x1[:, None] - x2[None, :]
    sq_dist = dx**2
    K = (stdev**2) * np.exp(-0.5 * sq_dist / (length_scale**2))
    dK_dx1 = K * (-dx) / (length_scale**2)
    return K, dK_dx1	
   

forrester_function = lambda x: (6 * x - 2 ) ** 2 * np.sin(12 * x - 4)
branin_function = lambda x: np.sin(3 * x) + 0.5 * x

class ExpectedImprovementRBFKernel:
	def __init__(self):
		pass

class AcquisitionFunction:
	def __init__(fn, deriv, prior_best):
		self.fn = fn
		self.deriv = deriv
		self.prior_best = prior_best
	
	def dmu_dx(alpha):
		self.fn = lambda dSigma_fy: dSigma_fy.T @ alpha
	
	def dsigma_dx(alpha):
		self.deriv
		
	def val(x):
		mu = self.mu(x)
		prior_best = self.prior_best
		sigma = self.posterior_covariance(x, x)
		return (mu - prior_best) * norm.cdf((
		

def bayesian_optimisation_1d(fn, lower_bound, upper_bound, kernel, max_iters=100, prior_mean=lambda x:0, starting_pts=5):
	if starting_pts:
		obvs_locations = np.random.uniform(lower_bound, upper_bound, starting_pts)
		obvs_values = np.array([fn(loc) for loc in obvs_locations])
	
	for iteration in range(max_iters):
		# Determine at which point to search next

		
		Sigma_yy, _ = kernel(observation_locations, observation_locations)
		jitter = 1e-8
		L = np.linalg.cholesky(Sigma_yy + jitter * np.eye(Sigma_yy.shape[0]))
		rhs = obvs_values
		
		
		alpha = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
		
		search_point = optimise_acquisition(acq, observations, lower_bound, upper_bound)
		obvs_locations = np.append(obvs_locations, search_point)
		obvs_values = np.append(obvs_values, fn(search_point))
		
		

		
		
		
		
		
		
def optimise_acquisition(acq, seeding_method = lambda x: np.random.uniform(x), num_seeds=100):
	best_result = 0
	seed_data = (lower_bound, upper_bound, num_seeds)
	seeds = seeding_method(**seed_data)
	
	for seed in seeds:
		success, val = gradient_ascent(acq.derivative, seed, lower_bound, upper_bound, rate=0.1)
		if success:
			if acq.fn(val) > best_result:
				best_location = val
				best_result = acq.fn(val)
				


def gradient_ascent(d_fn, seed, lower_bound, upper_bound, rate = 0.01, max_iters=100, tol=10e-6):
	x = seed
	high = False
	low = False
	for iteration in range(max_iters):
		x_new = x + rate * d_fn(x)
		
		if x > upper_bound:
			x = upper_bound
			if high:
				return False, "Out of Bounds"
			high = True
		elif x < lower_bound:
			x = lower_bound
			if low:
				return False, "Out of Bounds"
			low = True
		else:
			high = False
			low = False
		
		if abs(x_new - x) < tol:
			return True, x_new
		x = x_new
	return False, "Failed to Converge"
	

	
#bayesian_optimisation_1d(branin_function, 0, 5, rbf_kernel)
if __name__ == "__main__":
	print(gradient_descent(lambda x: 2*x, 0.5, -4, 4, rate=0.1))
	
