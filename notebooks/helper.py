import numpy as np
def gradient_descent(grad_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 - learning_rate * grad
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def newton_method(grad_func, hessian_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		# compute hessian
		hessian = hessian_func(w0, x_set, y_set)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# set serach direction by Newton method
		dx = np.linalg.solve(hessian, grad)
		# next step
		w1 = w0 - learning_rate * dx
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def bfgs_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	B0 = np.eye(len(w0))
	for i in range(MaxIter):
		grad = grad_func(w0,x_set, y_set)
		if np.linalg.norm(grad) < 1E-7:
			break
		p0 = -np.linalg.solve(B0, grad)
		s0 = learning_rate * p0
		w1 = w0 + s0
		y0 = (grad_func(w1) - grad).reshape(-1,1) # convert to a column vector
		B1 = B0 + np.dot(y0, y0.T) / np.dot(y0.T, s0) \
				- (np.dot(np.dot(B0, s0).reshape(-1,1), np.dot(s0, B0).reshape(-1,1).T)) / np.dot(np.dot(B0, s0), s0)
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
		B0 = B1
	return w0, path

def nelder_mead(func, x0):
    maxiter = 2000
    maxfun = None

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025

    dim = 5

    N = len(x0)

    sim = np.zeros((N + 1, N), dtype=x0.dtype)
    sim[0] = x0
    for k in range(N):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y

    fsim = np.zeros((N + 1,), float)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while iterations < maxiter :
        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in range(1, N + 1):
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        iterations += 1

    x = sim[0]
    fval = np.min(fsim)
    return x

def momentum_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, alpha=0.9, MaxIter=10):
	path = []
	path.append(w0)
	velocity = np.zeros_like(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		velocity = alpha * velocity  - learning_rate * grad
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 + velocity
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def nesterov_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, alpha=0.9, MaxIter=10):
	path = []
	path.append(w0)
	velocity = np.zeros_like(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0 + alpha * velocity, x_set, y_set)
		velocity = alpha * velocity  - learning_rate * grad
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 + velocity
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def adagrad_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, delta=1E-7, MaxIter=10):
	path = []
	path.append(w0)
	r = np.zeros_like(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		r = r  + grad * grad
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 - learning_rate * grad / ( delta + np.sqrt(r) )
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def rmsprop_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, delta=1E-6, rho=0.9 , MaxIter=10):
	path = []
	path.append(w0)
	r = np.zeros_like(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		r = rho * r  + (1 - rho) * (grad * grad)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 - learning_rate  * grad / np.sqrt(delta + r)
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def adam_method(grad_func, x_set, y_set, w0,
					learning_rate=0.001, delta=1E-8, rho1=0.9, rho2=0.999, MaxIter=10):
	path = []
	path.append(w0)
	s = np.zeros_like(w0)
	r = np.zeros_like(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		s = rho1 * s  + (1 - rho1) * grad
		r = rho2 * r  + (1 - rho2) * (grad * grad)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		shat = s / (1. - rho1 ** (i+1))
		rhat = r / (1. - rho2 ** (i+1))
		# next step
		w1 = w0 - learning_rate  * shat / (delta + np.sqrt(rhat))
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def generate_batches(batch_size, features, labels):
	"""
	Create batches of features and labels
	:param batch_size: The batch size
	:param features: List of features
	:param labels: List of labels
	:return: Batches of (Features, Labels)
	"""
	assert len(features) == len(labels)
	outout_batches = []

	sample_size = len(features)
	for start_i in range(0, sample_size, batch_size):
		end_i = start_i + batch_size
		batch = [features[start_i:end_i], labels[start_i:end_i]]
		outout_batches.append(batch)

	return outout_batches