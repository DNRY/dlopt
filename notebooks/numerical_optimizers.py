import numpy as np
def steepest_descent_2d(func, gradx, grady, x0, MaxIter=10, learning_rate=0.25, verbose=True):
    paths = [x0]
    fval_paths = [func(x0[0], x0[1])]
    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        x1 = x0 - learning_rate * grad
        fval = func(*x0)
        if verbose:
            print(i, x0, fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return(x0, fval, paths, fval_paths)

def newton_descent_2d(func, gradx, grady, hessian, x0, MaxIter=10, learning_rate=1.0, verbose=True):
    paths = [x0]
    fval_paths = [func(x0[0], x0[1])]
    for i in range(MaxIter):
        grad = np.array([gradx(*x0), grady(*x0)])
        hess = hessian(x0[0], x0[1])
        x1 = x0 - learning_rate * np.linalg.solve(hess, grad)
        fval = func(*x0)
        if verbose:
            print(i, x0, fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return(x0, fval, paths, fval_paths)