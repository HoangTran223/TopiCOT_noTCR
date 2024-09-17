import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

class NashMTL:
    def __init__(self):
        pass

    def compute_weights(self, grads):
        num_tasks = len(grads)
        #grads = [g.view(-1).cpu().numpy() for g in grads]
        grads = [g.view(-1).numpy() for g in grads]
        grads = np.stack(grads)  

        G = grads @ grads.T  

        w = cp.Variable(num_tasks)

        objective = cp.Minimize(0.5 * cp.quad_form(w, G))

        constraints = [w >= 0, cp.sum(w) == 1]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        weights = w.value
        return weights
