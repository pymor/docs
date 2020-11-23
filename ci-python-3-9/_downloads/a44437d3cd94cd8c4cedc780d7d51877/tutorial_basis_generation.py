#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import pymor.tools.random
pymor.tools.random._default_random_state = None


# In[2]:


import numpy as np
from pymor.basic import *


# In[3]:


problem = thermal_block_problem((3,3))
fom, _ = discretize_stationary_cg(problem, diameter=1/100)


# In[4]:


parameter_space = fom.parameters.space(0.0001, 1.)


# In[5]:


fom.parameters


# In[6]:


training_set = parameter_space.sample_randomly(25)
print(training_set)


# In[7]:


U = fom.solution_space.empty()
for mu in training_set:
    U.append(fom.solve(mu))


# In[8]:


fom.solution_space


# In[9]:


len(U)


# In[10]:


fom.visualize(U)


# In[11]:


trivial_basis = U.copy()


# In[12]:


V = fom.solve(parameter_space.sample_randomly(1)[0])


# In[13]:


G = trivial_basis.gramian()


# In[14]:


R = trivial_basis.inner(V)


# In[15]:


assert R.shape == (25,1)


# In[16]:


lambdas = np.linalg.solve(G, R)


# In[17]:


V_proj = trivial_basis.lincomb(lambdas.T)


# In[18]:


fom.visualize((V, V_proj, V - V_proj),
              legend=('V', 'V_proj', 'best-approximation err'),
              separate_colorbars=True)


# In[19]:


fom.h1_0_semi_product


# In[20]:


G = trivial_basis[:10].gramian(product=fom.h1_0_semi_product)
R = trivial_basis[:10].inner(V, product=fom.h1_0_semi_product)
lambdas = np.linalg.solve(G, R)
V_h1_proj = trivial_basis[:10].lincomb(lambdas.T)

fom.visualize((V, V_h1_proj, V - V_h1_proj), separate_colorbars=True)


# In[21]:


validation_set = parameter_space.sample_randomly(100)
V = fom.solution_space.empty()
for mu in validation_set:
    V.append(fom.solve(mu))


# In[22]:


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


# In[23]:


from matplotlib import pyplot as plt
plt.figure()
plt.semilogy(trivial_errors)
plt.ylim(1e-1, 1e1)
plt.show()


# In[24]:


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


# In[25]:


greedy_basis = strong_greedy(U, fom.h1_0_product, 25)


# In[26]:


greedy_errors = compute_proj_errors(greedy_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()


# In[27]:


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


# In[28]:


gram_schmidt(greedy_basis, product=fom.h1_0_semi_product, copy=False)
gram_schmidt(trivial_basis, product=fom.h1_0_semi_product, copy=False)


# In[29]:


G_trivial = trivial_basis.gramian(fom.h1_0_semi_product)
G_greedy = greedy_basis.gramian(fom.h1_0_semi_product)

print(f'trivial: {np.linalg.cond(G_trivial)}, '
      f'greedy: {np.linalg.cond(G_greedy)}')


# In[30]:


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


# In[31]:


pod_basis, pod_singular_values = pod(U, product=fom.h1_0_semi_product, modes=25)


# In[32]:


np.linalg.cond(pod_basis.gramian(fom.h1_0_semi_product))


# In[33]:


pod_errors = compute_proj_errors_orth_basis(pod_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.semilogy(pod_errors, label='POD')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()


# In[34]:


fom.visualize(pod_basis)


# In[35]:


reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)


# In[36]:


greedy_data = rb_greedy(fom, reductor, parameter_space.sample_randomly(1000),
                        max_extensions=25)


# In[37]:


weak_greedy_basis = reductor.bases['RB']


# In[38]:


weak_greedy_errors = compute_proj_errors_orth_basis(weak_greedy_basis, V, fom.h1_0_semi_product)

plt.figure()
plt.semilogy(trivial_errors, label='trivial')
plt.semilogy(greedy_errors, label='greedy')
plt.semilogy(pod_errors, label='POD')
plt.semilogy(weak_greedy_errors, label='weak greedy')
plt.ylim(1e-1, 1e1)
plt.legend()
plt.show()

