import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def rbf_kernel(x1, x2, length_scale=1.0, stdev=1.0):
    x1 = np.atleast_1d(x1).astype(float)
    x2 = np.atleast_1d(x2).astype(float)
    dx = x1[:, None] - x2[None, :]
    sq_dist = dx**2
    sigma = (stdev**2) * np.exp(-0.5 * sq_dist / (length_scale**2))
    dsigma_dx1 = sigma * (-dx) / (length_scale**2)
    return sigma, dsigma_dx1  # covariance and gradient wrt x1

forrester_function = lambda x: (6 * x - 2) ** 2 * np.sin(12 * x - 4)
branin_function    = lambda x: np.sin(3 * x) + 0.5 * x

class RBFKernel:
    def __init__(self, length_scale=0.5, stdev=1.0):
        self.length_scale = float(length_scale)
        self.stdev = float(stdev)
    def sigma(self, x1, x2):
        return rbf_kernel(x1, x2, self.length_scale, self.stdev)

class AcquisitionFunction:
    def __init__(self, kernel, prior_best, prior_mean=lambda x: 0.0):
        self.kernel = kernel
        self.prior_best = float(prior_best)
        self.prior_mean = prior_mean
        self.X = None
        self.L = None
        self.alpha = None

    def set_posterior(self, X, y):
        self.X = np.asarray(X, float)
        sigma_yy, _ = self.kernel.sigma(self.X, self.X)
        sigma_yy = sigma_yy + 1e-8 * np.eye(len(self.X))  # jitter for stability
        self.L = np.linalg.cholesky(sigma_yy)

        m_y = np.array([self.prior_mean(xi) for xi in self.X], dtype=float)
        rhs = y - m_y
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, rhs))

    def mu_sigma_and_grads(self, x):
        x = float(x)
        X = self.X
        L = self.L
        alpha = self.alpha
        st = self.kernel.stdev

        sigma_fy, dsigma_fy_dx = self.kernel.sigma(np.array([x]), X)
        sigma_fy = sigma_fy.ravel()      # (n,)
        dsigma_fy_dx = dsigma_fy_dx.ravel()

        # Posterior mean
        m_x = self.prior_mean(x)
        mu = m_x + sigma_fy @ alpha

        # Posterior variance
        v = np.linalg.solve(L, sigma_fy)     # L v = σ_fy
        sigma_ff = st**2 - v @ v
        sigma_ff = max(sigma_ff, 0.0)
        sigma = np.sqrt(sigma_ff)

        # Derivatives
        dmu_dx = dsigma_fy_dx @ alpha

        w = np.linalg.solve(L.T, v)          # σ_yy^{-1} σ_fy
        dsigma_ff_dx = -2.0 * (dsigma_fy_dx @ w)
        dsigma_dx = 0.5 * dsigma_ff_dx / sigma if sigma > 0.0 else 0.0

        return mu, sigma, dmu_dx, dsigma_dx

    def val(self, x):
        mu, sigma, _, _ = self.mu_sigma_and_grads(x)
        imp = mu - self.prior_best
        if sigma <= 0.0:
            return 0.0
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    def derivative(self, x):
        mu, sigma, dmu_dx, dsigma_dx = self.mu_sigma_and_grads(x)
        if sigma <= 0.0:
            return 0.0
        z = (mu - self.prior_best) / sigma
        return norm.cdf(z) * dmu_dx + norm.pdf(z) * dsigma_dx

def bayesian_optimisation_1d(fn, lower_bound, upper_bound, kernel, *,
                             max_iters=100, acquisition_seeds=100,
                             grad_descent_iters=100, grad_descent_rate=0.1,
                             starting_pts=5, prior_mean=lambda x: 0.0,
                             store_data=False):
    obvs_locations = np.random.uniform(lower_bound, upper_bound, starting_pts)
    obvs_values = np.array([fn(loc) for loc in obvs_locations], dtype=float)

    # --- store_data setup (create once) ---
    if store_data:
        X_domain = np.linspace(lower_bound, upper_bound, 101)
        data = {"posterior_means": [], "posterior_stdevs": [], "acquisition_vals": [], "obvs_values": [], "obvs_locations": []}
        data["X_vals"] = X_domain

    for _ in range(max_iters):
        if store_data:
            data["obvs_locations"].append(obvs_locations)
            data["obvs_values"].append(obvs_values)

        acq = AcquisitionFunction(kernel,
                                  prior_best=np.max(obvs_values),
                                  prior_mean=prior_mean)
        acq.set_posterior(obvs_locations, obvs_values)

        search_point = optimise_acquisition(
            acq, lower_bound, upper_bound,
            acquisition_seeds=acquisition_seeds,
            grad_descent_iters=grad_descent_iters,
            grad_descent_rate=grad_descent_rate
        )
        obvs_locations = np.append(obvs_locations, search_point)
        obvs_values = np.append(obvs_values, fn(search_point))

        if store_data:
            acquisition_vals = [acq.val(x) for x in X_domain]
            posterior_means  = [acq.mu_sigma_and_grads(x)[0] for x in X_domain]
            posterior_stdevs = [acq.mu_sigma_and_grads(x)[1] for x in X_domain]
            data["posterior_means"].append(posterior_means)
            data["posterior_stdevs"].append(posterior_stdevs)
            data["acquisition_vals"].append(acquisition_vals)

    if store_data:
        return obvs_locations, obvs_values, data

    return obvs_locations, obvs_values, None

def optimise_acquisition(acq, lower_bound, upper_bound, *,
                         seeding_method=None, acquisition_seeds=100,
                         grad_descent_iters=100, grad_descent_rate=0.1):
    if seeding_method is None:
        seeds = np.random.uniform(lower_bound, upper_bound, acquisition_seeds)
    else:
        seeds = seeding_method(lower_bound, upper_bound, acquisition_seeds)

    best_location = float(seeds[0])
    best_result = -np.inf

    for seed in seeds:
        success, x_star = gradient_ascent(
            acq.derivative, seed, lower_bound, upper_bound,
            rate=grad_descent_rate, max_iters=grad_descent_iters
        )
        fx = acq.val(x_star) if success else -math.inf
        if fx > best_result:
            best_result = fx
            best_location = x_star
    return best_location

def gradient_ascent(d_fn, seed, lower_bound, upper_bound, *,
                    rate=0.01, max_iters=100, tol=1e-6):
    x = float(seed)
    for _ in range(max_iters):
        x_new = x + rate * d_fn(x)
        if x_new > upper_bound:
            x_new = upper_bound
        if x_new < lower_bound:
            x_new = lower_bound
        if abs(x_new - x) < tol:
            return True, x_new
        x = x_new
    return True, x  # return last iterate

if __name__ == "__main__":
    np.random.seed(0)
    ker = RBFKernel(length_scale=0.5, stdev=1.0)

    lower_bound = 0.0
    upper_bound = 5.0

    X, y, data = bayesian_optimisation_1d(
        branin_function, lower_bound, upper_bound, ker,
        max_iters=20, starting_pts=6,
        acquisition_seeds=20, grad_descent_iters=20, grad_descent_rate=0.05,
        store_data=True
    )
    i_best = int(np.argmax(y))
    print(f"Best x ≈ {X[i_best]:.5f}, f(x) ≈ {y[i_best]:.5f}")

    xs = np.linspace(lower_bound, upper_bound, 400)
    fxs = np.array([branin_function(t) for t in xs])
    plt.plot(xs, fxs, label="f(x)")
    plt.scatter(X, y, s=30, zorder=3, label="samples")
    plt.axvline(X[i_best], ls="--", label="incumbent")
    plt.legend(); plt.title("1D BO (EI, maximize)"); plt.show()
