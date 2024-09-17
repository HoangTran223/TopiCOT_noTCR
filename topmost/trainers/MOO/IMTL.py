import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

class IMTL:
    def __init__(self):
        pass

    def compute_weights(self, grads):
        num_tasks = len(grads)
        grads = [g.view(-1) for g in grads]
        grads = torch.stack(grads) 

        G = grads @ grads.t() 

        G += 1e-8 * torch.eye(num_tasks).to(G.device)

        #G_np = G.cpu().numpy()
        G_np = G.numpy()
        ones = np.ones(num_tasks)

        weights = np.linalg.solve(G_np, ones)
        weights = weights / weights.sum()

        return weights
