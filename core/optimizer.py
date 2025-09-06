import numpy as np



def gd_optimizer(params, loss_grad_func, lr=0.01, epochs=1000, **kwargs):
    for _ in range(epochs):
        grads = loss_grad_func(params, **kwargs)
        for key in grads:
            params[key] -= lr * grads[key]
    return params


def sgd_optimizer(params, loss_grad_func, lr=0.01, epochs=1000, batch_size=1, **kwargs):
    pass
    return params


def qp_optimizer(params, loss_grad_func,**kwargs):
    pass
    return
    


def EM_optimizer(params, expectation_func, maximization_func):
    pass
    return 