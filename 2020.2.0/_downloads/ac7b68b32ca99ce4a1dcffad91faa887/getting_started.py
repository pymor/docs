#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pymor.basic import *
from pymor.core.logger import set_log_levels
set_log_levels({'pymor.algorithms.greedy': 'ERROR', 'pymor.algorithms.gram_schmidt.gram_schmidt': 'ERROR', 'pymor.algorithms.image.estimate_image_hierarchical': 'ERROR'})


# In[2]:


p = thermal_block_problem(num_blocks=(3, 2))


# In[3]:


fom, fom_data = discretize_stationary_cg(p, diameter=1./50.)


# In[4]:


print(fom_data['grid'])


# In[5]:


U = fom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
fom.visualize(U, title='Solution')


# In[6]:


print(fom.parameters)


# In[7]:


reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)


# In[8]:


training_set = p.parameter_space.sample_uniformly(4)
print(training_set[0])


# In[9]:


greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=32)


# In[10]:


print(greedy_data.keys())


# In[11]:


rom = greedy_data['rom']


# In[12]:


RB = reductor.bases['RB']
print(type(RB))
print(len(RB))
print(RB.dim)


# In[13]:


import numpy as np
gram_matrix = RB.gramian(fom.h1_0_semi_product)
print(np.max(np.abs(gram_matrix - np.eye(32))))


# In[14]:


u = rom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
print(u)
U_red = reductor.reconstruct(u)
print(U_red.dim)


# In[15]:


ERR = U - U_red
print(ERR.norm(fom.h1_0_semi_product))
fom.visualize((U, U_red, ERR),
              legend=('Detailed', 'Reduced', 'Error'),
              separate_colorbars=True)

