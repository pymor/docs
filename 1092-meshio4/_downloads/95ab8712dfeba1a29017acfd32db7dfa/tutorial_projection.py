#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')


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


print_source(fom._solve)


# In[8]:


type(fom)


# In[9]:


fom.rhs


# In[10]:


fom.rhs.source


# In[11]:


from pymor.operators.interface import Operator
print_source(Operator.as_range_array)


# In[12]:


U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=[1., 0.1, 0.1, 1.])


# In[13]:


mu = fom.parameters.parse([1., 0.1, 0.1, 1.])
U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=mu)


# In[14]:


(U-U2).norm()


# In[15]:


reduced_operator = fom.operator.apply2(basis, basis, mu=mu)
reduced_rhs = basis.inner(fom.rhs.as_range_array(mu))


# In[16]:


import numpy as np

u_N = np.linalg.solve(reduced_operator, reduced_rhs)
u_N


# In[17]:


U_N = basis.lincomb(u_N.T)
U_N


# In[18]:


(U-U_N).norm(fom.h1_0_product) / U.norm(fom.h1_0_product)


# In[19]:


fom.visualize((U, U_N, U-U_N), separate_colorbars=True)


# In[20]:


type(reduced_operator)


# In[21]:


from pymor.operators.numpy import NumpyMatrixOperator

reduced_operator = NumpyMatrixOperator(reduced_operator)
reduced_rhs = NumpyMatrixOperator(reduced_rhs)


# In[22]:


from pymor.models.basic import StationaryModel
rom = StationaryModel(reduced_operator, reduced_rhs)
rom


# In[23]:


u_N2 = rom.solve()
u_N.T - u_N2.to_numpy()


# In[24]:


print(fom.parameters)
print(rom.parameters)


# In[25]:


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


# In[26]:


fom.operator


# In[27]:


reduced_operators = [NumpyMatrixOperator(op.apply2(basis, basis))
                     for op in fom.operator.operators]


# In[28]:


reduced_operator = fom.operator.with_(operators=reduced_operators)


# In[29]:


fom.rhs.parameters


# In[30]:


rom = StationaryModel(reduced_operator, reduced_rhs)


# In[31]:


rom.parameters


# In[32]:


u_N3 = rom.solve(mu)
u_N.T - u_N3.to_numpy()


# In[33]:


tic = perf_counter()
fom.solve(mu)
toc = perf_counter()
rom.solve(mu)
tac = perf_counter()
print(f'FOM: {toc-tic:.5f} (s)')
print(f'ROM: {tac-toc:.5f} (s)')


# In[34]:


from pymor.algorithms.projection import project

reduced_operator = project(fom.operator, basis, basis)
reduced_rhs      = project(fom.rhs,      basis, None )


# In[35]:


reduced_operator


# In[36]:


rom = StationaryModel(reduced_operator, reduced_rhs)
u_N4 = rom.solve(mu)
u_N.T - u_N4.to_numpy()


# In[37]:


print_source(project)


# In[38]:


from pymor.algorithms.projection import ProjectRules
ProjectRules


# In[39]:


assert ProjectRules.rules[8].action_description == 'LincombOperator'


# In[40]:


ProjectRules.rules[8]


# In[41]:


assert ProjectRules.rules[3].action_description == 'apply_basis'


# In[42]:


ProjectRules.rules[3]


# In[43]:


from pymor.reductors.basic import StationaryRBReductor

reductor = StationaryRBReductor(fom, basis)
rom = reductor.reduce()


# In[44]:


u_N5 = rom.solve(mu)
u_N.T - u_N5.to_numpy()


# In[45]:


print_source(reductor.project_operators)


# In[46]:


print_source(reductor.build_rom)


# In[47]:


U_N5 = reductor.reconstruct(u_N5)
(U_N - U_N5).norm()


# In[48]:


print_source(reductor.reconstruct)

