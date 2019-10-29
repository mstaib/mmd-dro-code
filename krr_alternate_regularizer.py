import numpy as np

from scipy.optimize import minimize


def fn_and_grad_square_reg(d, K):
    # auto-generated from matrixcalculus.org

    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    assert isinstance(d, np.ndarray)
    dim = d.shape
    assert len(dim) == 1
    d_rows = dim[0]
    assert K_cols == d_rows == K_rows

    t_0 = (1 / 2)
    T_1 = (d[:, np.newaxis] * K)
    functionValue = (np.trace((T_1 ** 4)) ** t_0)
    gradient = (((4 * (np.trace(((K * d[np.newaxis, :]) ** 4)) ** (t_0 - 1))) / 2) * np.diag(np.dot((np.eye(d_rows, K_cols) * (T_1 ** 3)), K)))

    return functionValue, gradient



def fn_and_grad_normal_reg(a, K):
    n = len(a)
    
    reg = a.T.dot(K.dot(a))
    grad = 2 * K.dot(a)

    return reg, grad



def fit_krr_both_regularizers(K, y, lmb_new, lmb_old, a_init=None):
    def fun(a):
        errs = K.dot(a) - y

        val_fit = np.sum(errs ** 2)
        grad_fit = 2 * K.dot(errs)

        val_total = val_fit 
        grad_total = grad_fit

        if lmb_new > 0:
            if np.all(a == 0):
                val_reg_new = 0
                grad_reg_new = np.zeros(grad_total.shape)
            else:
                val_reg_new, grad_reg_new = fn_and_grad_square_reg(a, K ** 0.5)

            val_total += lmb_new * val_reg_new
            grad_total += lmb_new * grad_reg_new

        if lmb_old > 0:
            # val_reg_old = a.T.dot(K.dot(a))
            # grad_reg_old = 2*K.dot(a)

            val_reg_old, grad_reg_old = fn_and_grad_normal_reg(a, K)

            val_total += lmb_old * val_reg_old
            grad_total += lmb_old * grad_reg_old

        return val_total, grad_total
    
    if a_init is None:
        a_init = np.zeros(len(y))
    res = minimize(fun, 
                   a_init, 
                   method='L-BFGS-B', 
                   jac=True,
                   options={'gtol': 1e-8, 'ftol': 1e-10})


    
    val = res.fun
    a = res.x
    return a, val



def create_population_distribution_2(seed=1, n=10000):
    np.random.seed(seed)

    k = 2
    true_x = np.reshape(np.array([-1,1]), (k,1))
    true_a = np.reshape(np.array([-1,1]), (1,k))

    from sklearn.gaussian_process.kernels import RBF
    m = RBF()

    def true_eval(x_new):
        return np.sum(m(x_new, true_x) * true_a, axis=1)

    x = np.random.randn(n, 1)
    x = np.array(sorted(x))

    # noise = 1*np.random.randn(len(x),1)
    # y = np.array([true_eval(x[inx]) + noise[inx] for inx in range(len(x))]).flatten()
    y = true_eval(x).flatten()
    # y = np.array([true_eval(x[inx]) for inx in range(len(x))]).flatten()

    return x, y, m, true_eval


def sample_from_population(x, y, n=20, stddev=1., seed=1):
    np.random.seed(seed)

    sample_inx = np.random.choice(range(len(x)), n)

    return x[sample_inx], y[sample_inx] + stddev*np.random.randn(n)



def create_pred(x, m):
    def pred(x_new, a_learned):
        return np.sum( m(x_new, x) * a_learned, axis=1)

    return pred

def create_fit_quality(pred, x, y):
    def fit_quality(a_learned):
        y_fit = pred(x, a_learned)

        return np.mean((y_fit - y) ** 2)

    return fit_quality


def main(seed1, seed2, lmbs_old, lmbs_new, n_sample=20, stddev=1., verbose=False):
    x, y, m, true_eval = create_population_distribution_2(seed1)

    x_s, y_s = sample_from_population(x, y, n_sample, seed=seed2, stddev=stddev)
    K = m(x_s)
    pred = create_pred(x_s, m)

    fit_quality = create_fit_quality(pred, x, y)

    out_dicts = []

    a_init, val = fit_krr_both_regularizers(K, y_s, 0, 0)

    for lmb_old in lmbs_old:
        for lmb_new in lmbs_new:
            a, val = fit_krr_both_regularizers(K, y_s, lmb_new, lmb_old, a_init)

            this_dict = {
                'lmb_old': lmb_old,
                'lmb_new': lmb_new,
                'l2': fit_quality(a)
            }

            out_dicts.append(this_dict)
            if verbose:
                print(this_dict)


    return {
        'lmbs_new': lmbs_new, 
        'lmbs_old': lmbs_old, 
        'out_dicts': out_dicts
    }