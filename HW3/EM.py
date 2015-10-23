import numpy as np
import numpy.random as random
import scipy.sparse as sparse
import math

def em(x, k):
    # r = initialization(x, k)
    r = my_init(x, k)
    label = np.argmax(r, 1)
    u = np.unique(label)
    r = remove_empty_component(r, u)

    tol = 1e-10
    maxiter = 500
    llh = []
    for i in range(maxiter):
        llh.append(-float('inf'))
    converged = False
    t = 0
    model = None
    while not converged and t < maxiter:
        t += 1
        model = maximization(x, r)
        r, llh[t] = expectation(x, model)
        label = np.argmax(r, 1)
        u = np.unique(label)
        if len(r[0]) != len(u):     # TODO confirm the axis
            r = remove_empty_component(r, u)
        else:
            converged = llh[t] - llh[t-1] < tol*abs(llh[t])
    llh = llh[1:t]
    if converged:
        print 'Converged in {} steps.\n'.format(str(t))
    else:
        print 'Not converged in {} steps.\n'.format(str(maxiter))
    return label, model, llh


def remove_empty_component(r, u):
    u = np.sort(u)
    res = []
    for i, rr in enumerate(r):
        res.append([])
        for uu in u:
            res[i].append(rr[uu])
    return res

def expectation(x, model):
    mu = model['mu']
    sigma = model['sigma']
    w = model['weight']

    n = np.shape(x)[1]
    k = np.shape(mu)[1]
    log_rho = np.zeros((n, k))

    for i in range(k):
        log_rho[:,i] = log_gauss_pdf(x, mu[:,i], sigma[:,:,i])
    log_rho = log_rho + np.log(w)
    t = log_sum_exp(log_rho, 1)
    llh = sum(t) / n
    log_r = []
    for i in range(n):  # log_rho - t
        log_r.append(log_rho[i] - t[i])
    log_r = np.array(log_r)
    r = np.exp(log_r)
    return r, llh



# function s = logsumexp(x, dim)
# % Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
# %   By default dim = 1 (columns).
# % Written by Michael Chen (sth4nth@gmail.com).
# if nargin == 1,
#     % Determine which dimension sum will use
#     dim = find(size(x)~=1,1);
#     if isempty(dim), dim = 1; end
# end
#
# % subtract the largest in each column
# y = max(x,[],dim);
# x = bsxfun(@minus,x,y);
# s = y + log(sum(exp(x),dim));
# i = find(~isfinite(y));
# if ~isempty(i)
#     s(i) = y(i);
# end

def log_sum_exp(x, dim):
    n, d = np.shape(x)
    y = np.max(x, dim)
    x_tmp = []
    for i in range(n):
        x_tmp.append(x[i] - y[i])
    x_tmp = np.array(x_tmp)
    s = y + np.log(np.sum(np.exp(x_tmp), dim))
    # TODO handle underflow
    # i = np.isfinite()

    return s


def log_gauss_pdf(x, mu, sigma):
    d = len(x)
    x = x - mu
    u = np.linalg.cholesky(sigma)
    q = np.linalg.solve(np.transpose(u), x)
    qq = my_dot(q)
    c = d * math.log(2 * math.pi) + 2 * np.sum(np.log(np.diag(u)))
    y = -(c + qq) / 2.
    return np.matrix(y)

def maximization(x, r):
    model = {}
    d, n = np.shape(x)
    k = np.shape(r)[1]

    nk = np.sum(r, 0)
    w = nk / n
    mu = np.dot(x, r) / nk

    sigma = np.zeros((d, d, k))
    sqrt_r = np.sqrt(r)
    for i in range(k):
        xo = np.array(x - mu[:,i])
        xo = xo * np.array(sqrt_r[:,i])
        sigma[:,:,i] = np.dot(xo, np.transpose(xo)) / nk[i]
        sigma[:,:,i] = sigma[:,:,i] + np.eye(d) * (1e-6)

    model['mu'] = mu
    model['sigma'] = sigma
    model['weight'] = w
    return model

def initialization(x, k):
    d, n = np.shape(x)
    # TODO make sure x don't need to be convert to np.matrix
    idx = random.random_integers(1, n, (k, 1))
    m = [[xx[i] for xx in x] for i in idx]
    m = np.matrix(m)
    tmp1 = np.dot(np.transpose(m), x)
    tmp2 = [np.dot(d, d) for d in [[mm[i] for mm in m] for i in range(d)]]
    tmp = tmp1 - np.transpose(tmp2) / 2
    label = np.argmax(tmp, 0).tolist()
    # TODO check label need to be array
    u, label = np.unique(label, return_inverse=True)
    while k != len(u):
        idx = random.random_integers(1, n, (k, 1))
        m = [[xx[i] for xx in x] for i in idx]
        m = np.matrix(m)
        tmp1 = np.dot(np.transpose(m), x)
        tmp2 = my_dot(m)
        tmp = tmp1 - np.transpose(tmp2)
        label = np.argmax(tmp, 0).tolist()
        # TODO check label need to be array
        u, label = np.unique(label, return_inverse=True)
    # r = sparse.
    pass

def my_dot(m):
    d, n = np.shape(m)
    m_list = m.tolist()
    res = []
    for i in range(n):
        tmp = []
        for j in range(d):
            tmp.append(m_list[j][i])
        res.append(np.dot(tmp, tmp))

    # return [np.dot(c, c) for c in [[mm[i] for mm in m_tmp] for i in range(d)]]
    return res


def my_init(x, k):
    d, n = np.shape(x)
    r = np.zeros((n, k))
    for i in range(n):
        rnd = random.randint(0, k)
        r[i][rnd] = 1
    return r