%matplotlib inline

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')

import numpy as np
from pymor.basic import *

problem = thermal_block_problem((3,3))
fom, _ = discretize_stationary_cg(problem, diameter=1/100)

parameter_space = fom.parameters.space(0.0001, 1.)

fom.parameters

training_set = parameter_space.sample_randomly(25)
print(training_set)

U = fom.solution_space.empty()
for mu in training_set:
    U.append(fom.solve(mu))

fom.solution_space

len(U)

fom.visualize(U)

trivial_basis = U.copy()

V = fom.solve(parameter_space.sample_randomly(1)[0])

G = trivial_basis.gramian()

R = trivial_basis.inner(V)

assert R.shape == (25,1)

lambdas = np.linalg.solve(G, R)

V_proj = trivial_basis.lincomb(lambdas.T)

fom.visualize((V, V_proj, V - V_proj),
              legend=('V', 'V_proj', 'best-approximation err'),
              separate_colorbars=True)

fom.h1_0_semi_product

G = trivial_basis[:10].gramian(product=fom.h1_0_semi_product)
R = trivial_basis[:10].inner(V, product=fom.h1_0_semi_product)
lambdas = np.linalg.solve(G, R)
V_h1_proj = trivial_basis[:10].lincomb(lambdas.T)

fom.visualize((V, V_h1_proj, V - V_h1_proj), separate_colorbars=True)

validation_set = parameter_space.sample_randomly(100)
V = fom.solution_space.empty()
for mu in validation_set:
    V.append(fom.solve(mu))

def compute_proj_errors(basis, V, product):
    G = basis.gramian(product=product)
    R = basis.inner(V, product=product)
    errors = []
    for N in range(len(basis) + 1):
        if N > 0:
            v = np.linalg.solve(G[:N, :N], R[:N, :])
        else:
            v = np.zeros((0, len(V)))
        V_proj = basis[:N].lincomb(v.T)
        errors.append(np.max((V - V_proj).norm(product=product)))
    return errors

trivial_errors = compute_proj_errors(trivial_basis, V, fom.h1_0_semi_product)

from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(trivial_errors)
plt.ylim(1e-1, 1e1)
plt.show()

def strong_greedy(U, product, N):
    basis = U.space.empty()

    for n in range(N):
        # compute projection errors
        G = basis.gramian(product)
        R = basis.inner(U, product=product)
        lambdas = np.linalg.solve(G, R)
        U_proj = basis.lincomb(lambdas.T)
        errors = (U - U_proj).norm(product)

        # extend basis
        basis.append(U[np.argmax(errors)])

    return basis

greedy_basis = strong_greedy(U, fom.h1_0_product, 25)

greedy_errors = compute_proj_errors(greedy_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()

G_trivial = trivial_basis.gramian(fom.h1_0_semi_product)
G_greedy = greedy_basis.gramian(fom.h1_0_semi_product)
trivial_conds, greedy_conds = [], []
for N in range(1, len(U)):
    trivial_conds.append(np.linalg.cond(G_trivial[:N, :N]))
    greedy_conds.append(np.linalg.cond(G_greedy[:N, :N]))
plt.figure()
plt.semilogy(range(1, len(U)), trivial_conds, label='trivial')
plt.semilogy(range(1, len(U)), greedy_conds, label='greedy')
plt.legend()
plt.show()

gram_schmidt(greedy_basis, product=fom.h1_0_semi_product, copy=False)
gram_schmidt(trivial_basis, product=fom.h1_0_semi_product, copy=False)

G_trivial = trivial_basis.gramian(fom.h1_0_semi_product)
G_greedy = greedy_basis.gramian(fom.h1_0_semi_product)

print(f'trivial: {np.linalg.cond(G_trivial)}, '
      f'greedy: {np.linalg.cond(G_greedy)}')

def compute_proj_errors_orth_basis(basis, V, product):
    errors = []
    for N in range(len(basis) + 1):
        v = V.inner(basis[:N], product=product)
        V_proj = basis[:N].lincomb(v)
        errors.append(np.max((V - V_proj).norm(product)))
    return errors

trivial_errors = compute_proj_errors_orth_basis(trivial_basis, V, fom.h1_0_semi_product)
greedy_errors  = compute_proj_errors_orth_basis(greedy_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()

pod_basis, pod_singular_values = pod(U, product=fom.h1_0_semi_product, modes=25)

np.linalg.cond(pod_basis.gramian(fom.h1_0_semi_product))

pod_errors = compute_proj_errors_orth_basis(pod_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.semilogy(pod_errors, label='POD')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()

fom.visualize(pod_basis)

reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)

greedy_data = rb_greedy(fom, reductor, parameter_space.sample_randomly(1000),
                        max_extensions=25)

weak_greedy_basis = reductor.bases['RB']

weak_greedy_errors = compute_proj_errors_orth_basis(weak_greedy_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.semilogy(pod_errors, label='POD')
plt.semilogy(weak_greedy_errors, label='weak greedy')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()