%matplotlib inline

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor

k = 50
n = 2 * k + 1

A = sps.diags(
    [(n - 1) * [(n - 1)**2], n * [-2 * (n - 1)**2], (n - 1) * [(n - 1)**2]],
    [-1, 0, 1],
    format='lil',
)
A[0, 0] = A[-1, -1] = -2 * n * (n - 1)
A[0, 1] = A[-1, -2] = 2 * (n - 1)**2
A = A.tocsc()

B = np.zeros((n, 2))
B[:, 0] = 1
B[0, 1] = 2 * (n - 1)

C = np.zeros((3, n))
C[0, 0] = C[1, k] = C[2, -1] = 1

fom = LTIModel.from_matrices(A, B, C)

fom

print(fom)

w = np.logspace(-2, 8, 50)
fom.mag_plot(w)
plt.grid()

hsv = fom.hsv()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
ax.set_title('Hankel singular values')
ax.grid()

bt = BTReductor(fom)

error_bounds = bt.error_bounds()
fig, ax = plt.subplots()
ax.semilogy(range(1, len(error_bounds) + 1), error_bounds, '.-')
ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
ax.set_xlabel('Reduced order')
ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds')
ax.grid()

rom = bt.reduce(10)

fig, ax = plt.subplots()
fom.mag_plot(w, ax=ax, label='FOM')
rom.mag_plot(w, ax=ax, linestyle='--', label='ROM')
ax.legend()
ax.grid()

(fom - rom).mag_plot(w)
plt.grid()

print(f'Relative Hinf error: {(fom - rom).hinf_norm() / fom.hinf_norm():.3e}')
print(f'Relative H2 error:   {(fom - rom).h2_norm() / fom.h2_norm():.3e}')