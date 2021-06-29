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


import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

plt.rcParams['axes.grid'] = True


# In[3]:


k = 50
n = 2 * k + 1

E = sps.eye(n, format='lil')
E[0, 0] = E[-1, -1] = 0.5
E = E.tocsc()

d0 = n * [-2 * (n - 1)**2]
d1 = (n - 1) * [(n - 1)**2]
A = sps.diags([d1, d0, d1], [-1, 0, 1], format='lil')
A[0, 0] = A[-1, -1] = -n * (n - 1)
A = A.tocsc()

B = np.zeros((n, 2))
B[:, 0] = 1
B[0, 0] = B[-1, 0] = 0.5
B[0, 1] = n - 1

C = np.zeros((3, n))
C[0, 0] = C[1, k] = C[2, -1] = 1


# In[4]:


fom = LTIModel.from_matrices(A, B, C, E=E)


# In[5]:


fom


# In[6]:


print(fom)


# In[7]:


from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
fom = fom.with_(T=10, time_stepper=ImplicitEulerTimeStepper(100))


# In[8]:


u = lambda t: np.array([[np.sin(t)], [np.sin(2 * t)]])
Y = fom.output(input=u)
fig, ax = plt.subplots()
ax.plot(np.linspace(0, fom.T, fom.time_stepper.nt + 1), Y)
ax.set_xlabel('$t$')
ax.set_ylabel('$y(t)$')
_ = ax.set_title('Output')


# In[9]:


print(fom.eval_tf(0))
print(fom.eval_tf(1))
print(fom.eval_tf(1j))


# In[10]:


print(fom.eval_dtf(0))
print(fom.eval_dtf(1))
print(fom.eval_dtf(1j))


# In[11]:


w = np.logspace(-2, 8, 300)
_ = fom.mag_plot(w)


# In[12]:


_ = fom.bode_plot(w)


# In[13]:


poles = fom.poles()
fig, ax = plt.subplots()
ax.plot(poles.real, poles.imag, '.')
_ = ax.set_title('Poles')


# In[14]:


fom.gramian('c_lrcf')


# In[15]:


hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
_ = ax.set_title('Hankel singular values')


# In[16]:


fom.h2_norm()


# In[17]:


fom.hinf_norm()


# In[18]:


fom.hankel_norm()

