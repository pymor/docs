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


from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg

p = thermal_block_problem((2,2))
fom, _ = discretize_stationary_cg(p, diameter=1/100)


# In[3]:


U = fom.solve([1., 0.1, 0.1, 1.])
fom.visualize(U)


# In[4]:


from pymor.algorithms.pod import pod
from matplotlib import pyplot as plt

snapshots = fom.solution_space.empty()
for mu in p.parameter_space.sample_randomly(20):
    snapshots.append(fom.solve(mu))
basis, singular_values = pod(snapshots, modes=10)


# In[5]:


_ = plt.semilogy(singular_values)


# In[6]:


from pymor.tools.formatsrc import print_source
print_source(fom.solve)


# In[7]:


print_source(fom.compute)


# In[8]:


print_source(fom._compute_solution)


# In[9]:


type(fom)


# In[10]:


fom.rhs


# In[11]:


fom.rhs.source


# In[12]:


from pymor.operators.interface import Operator
print_source(Operator.as_range_array)


# In[13]:


U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=[1., 0.1, 0.1, 1.])


# In[14]:


mu = fom.parameters.parse([1., 0.1, 0.1, 1.])
U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=mu)


# In[15]:


(U-U2).norm()


# In[16]:


reduced_operator = fom.operator.apply2(basis, basis, mu=mu)
reduced_rhs = basis.inner(fom.rhs.as_range_array(mu))


# In[17]:


import numpy as np

u_N = np.linalg.solve(reduced_operator, reduced_rhs)
u_N


# In[18]:


U_N = basis.lincomb(u_N.T)
U_N


# In[19]:


(U-U_N).norm(fom.h1_0_product) / U.norm(fom.h1_0_product)


# In[20]:


fom.visualize((U, U_N, U-U_N), separate_colorbars=True)


# In[21]:


type(reduced_operator)


# In[22]:


from pymor.operators.numpy import NumpyMatrixOperator

reduced_operator = NumpyMatrixOperator(reduced_operator)
reduced_rhs = NumpyMatrixOperator(reduced_rhs)


# In[23]:


from pymor.models.basic import StationaryModel
rom = StationaryModel(reduced_operator, reduced_rhs)
rom


# In[24]:


u_N2 = rom.solve()
u_N.T - u_N2.to_numpy()


# In[25]:


print(fom.parameters)
print(rom.parameters)


# In[26]:


from time import perf_counter

tic = perf_counter()
fom.solve(mu)
toc = perf_counter()
fom.operator.apply2(basis, basis, mu=mu)
basis.inner(fom.rhs.as_range_array(mu))
tac = perf_counter()
rom.solve()
tuc = perf_counter()
print(f'FOM:          {toc-tic:.5f} (s)')
print(f'ROM assemble: {tac-toc:.5f} (s)')
print(f'ROM solve:    {tuc-tac:.5f} (s)')


# In[27]:


fom.operator


# In[28]:


reduced_operators = [NumpyMatrixOperator(op.apply2(basis, basis))
                     for op in fom.operator.operators]


# In[29]:


reduced_operator = fom.operator.with_(operators=reduced_operators)


# In[30]:


fom.rhs.parameters


# In[31]:


rom = StationaryModel(reduced_operator, reduced_rhs)


# In[32]:


rom.parameters


# In[33]:


u_N3 = rom.solve(mu)
u_N.T - u_N3.to_numpy()


# In[34]:


tic = perf_counter()
fom.solve(mu)
toc = perf_counter()
rom.solve(mu)
tac = perf_counter()
print(f'FOM: {toc-tic:.5f} (s)')
print(f'ROM: {tac-toc:.5f} (s)')


# In[35]:


from pymor.algorithms.projection import project

reduced_operator = project(fom.operator, basis, basis)
reduced_rhs      = project(fom.rhs,      basis, None )


# In[36]:


reduced_operator


# In[37]:


rom = StationaryModel(reduced_operator, reduced_rhs)
u_N4 = rom.solve(mu)
u_N.T - u_N4.to_numpy()


# In[38]:


print_source(project)


# In[39]:


from pymor.algorithms.projection import ProjectRules
ProjectRules


# In[40]:


assert ProjectRules.rules[8].action_description == 'LincombOperator'


# In[41]:


ProjectRules.rules[8]


# In[42]:


assert ProjectRules.rules[3].action_description == 'apply_basis'


# In[43]:


ProjectRules.rules[3]


# In[44]:


from pymor.reductors.basic import StationaryRBReductor

reductor = StationaryRBReductor(fom, basis)
rom = reductor.reduce()


# In[45]:


u_N5 = rom.solve(mu)
u_N.T - u_N5.to_numpy()


# In[46]:


print_source(reductor.project_operators)


# In[47]:


print_source(reductor.build_rom)


# In[48]:


U_N5 = reductor.reconstruct(u_N5)
(U_N - U_N5).norm()


# In[49]:


print_source(reductor.reconstruct)

