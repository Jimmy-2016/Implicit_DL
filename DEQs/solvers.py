from model import *


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res



# class Anderson:
#     def __init__(self, depth=5, dim=10, tol=1e-4, max_iter=50):
#         self.depth = depth
#         self.dim = dim
#         self.tol = tol
#         self.max_iter = max_iter
#
#     def solve(self, function, x):
#         # Memory for Anderson
#         m = min(self.depth, self.max_iter)
#         X = torch.zeros(m+1, *x.shape, device=x.device, dtype=x.dtype)
#         F = torch.zeros(m+1, *x.shape, device=x.device, dtype=x.dtype)
#
#         # Initial guess
#         X[0], F[0] = x, function(x)
#         x = X[0] - F[0]
#
#         for k in range(1, self.max_iter):
#             X[k % (m+1)] = x
#             F[k % (m+1)] = function(x)
#             # Solve the least squares problem to find the coefficients
#             if k > 1:
#                 # Form the matrix of differences
#                 n = min(k, m)
#                 G = F[:n+1] - X[:n+1]
#                 GTG = torch.matmul(G.T, G)
#                 GTg = torch.matmul(G.T, F[k % (m+1)] - X[k % (m+1)])
#                 alpha = torch.linalg.solve(GTG, GTg)
#                 # Compute the next iterate
#                 x = (X[:n+1] - torch.matmul(G, alpha)).mean(dim=0)
#             else:
#                 x = X[k % (m+1)] - F[k % (m+1)]
#
#             # Check for convergence
#             if torch.norm(F[k % (m+1)]) < self.tol:
#                 break
#
#         return x


# def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode='rel', beta=1.0, **kwargs):
#     """ Anderson acceleration for fixed point iteration. """
#     bsz, d, L = x0.shape
#     alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
#     X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
#     F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
#     X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
#     X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)
#
#     H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
#     H[:, 0, 1:] = H[:, 1:, 0] = 1
#     y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
#     y[:, 0] = 1
#
#     trace_dict = {'abs': [],
#                   'rel': []}
#     lowest_dict = {'abs': 1e8,
#                    'rel': 1e8}
#     lowest_step_dict = {'abs': 0,
#                         'rel': 0}
#
#     for k in range(2, threshold):
#         n = min(k, m)
#         G = F[:, :n] - X[:, :n]
#         H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
#             None]
#         alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
#
#         X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
#         F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
#         gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
#         abs_diff = gx.norm().item()
#         rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
#         diff_dict = {'abs': abs_diff,
#                      'rel': rel_diff}
#         trace_dict['abs'].append(abs_diff)
#         trace_dict['rel'].append(rel_diff)
#
#         for mode in ['rel', 'abs']:
#             if diff_dict[mode] < lowest_dict[mode]:
#                 if mode == stop_mode:
#                     lowest_xest, lowest_gx = X[:, k % m].view_as(x0).clone().detach(), gx.clone().detach()
#                 lowest_dict[mode] = diff_dict[mode]
#                 lowest_step_dict[mode] = k
#
#         if trace_dict[stop_mode][-1] < eps:
#             for _ in range(threshold - 1 - k):
#                 trace_dict[stop_mode].append(lowest_dict[stop_mode])
#                 trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
#             break
#
#     out = {"result": lowest_xest,
#            "lowest": lowest_dict[stop_mode],
#            "nstep": lowest_step_dict[stop_mode],
#            "prot_break": False,
#            "abs_trace": trace_dict['abs'],
#            "rel_trace": trace_dict['rel'],
#            "eps": eps,
#            "threshold": threshold}
#     X = F = None
#     return out



