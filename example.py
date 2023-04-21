import numpy as np
from glassobind import glasso


# Generate synthetic data
rng = np.random.default_rng(0)
d = 3
X = rng.normal(size=(128, d))

# Prepare inputs for graphical lasso
emp_cov = np.cov(X.T)
lmb = np.ones((d, d)) * .1



print('==================')
print('Run graphical lasso')
print('==================', end='\n\n')

print('emp_cov:\n', emp_cov, end='\n\n')
print('lmb:\n', lmb, end='\n\n')

result = glasso(
        emp_cov=emp_cov,        # Empirical covariance of shape (d, d)
        lmb=lmb,                # Element-wise penalty for graphical lasso of shape (d, d)
        sigma_init=np.eye(d),   # Initial guess of sigma (population covariance matrix)
        theta_init=np.eye(d),   # Initial guess of theta (population precision matrix)
        tol=1e-5,               # Tolerance of dual gap
        iter_max=100,           # Maximum iteration
        verbose=True,           # If True, print iteration-wise debug info
        eps=1e-8,               # Small number of numeric stability
        )
print('\n')

print('==================')
print('Result')
print('==================', end='\n\n')


print('theta (estimated precision):\n', result.theta, end='\n\n')
print('sigma (estimated covariance):\n', result.sigma, end='\n\n')
print('converged:\n', result.converged, end='\n\n')


print('==================')
print('Reusing previous result...')
print('Should stop on the first iteration because we did not change the other parameters.')
print('==================', end='\n\n')
result = glasso(
        emp_cov=emp_cov,
        lmb=lmb,
        sigma_init=result.sigma,  # reuse the previous result
        theta_init=result.theta,  # reuse the previous result
        tol=1e-5,
        iter_max=100,
        verbose=True,
        eps=1e-8,
        )
print('\n')
