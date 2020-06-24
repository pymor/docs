from pymor.basic import *
from pymor.core.logger import set_log_levels
set_log_levels({'pymor.algorithms.greedy': 'ERROR', 'pymor.algorithms.gram_schmidt.gram_schmidt': 'ERROR', 'pymor.algorithms.image.estimate_image_hierarchical': 'ERROR'})

p = thermal_block_problem(num_blocks=(3, 2))

fom, fom_data = discretize_stationary_cg(p, diameter=1./50.)

print(fom_data['grid'])

U = fom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
fom.visualize(U, title='Solution')

print(fom.parameters)

reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)

training_set = p.parameter_space.sample_uniformly(4)
print(training_set[0])

greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=32)

print(greedy_data.keys())

rom = greedy_data['rom']

RB = reductor.bases['RB']
print(type(RB))
print(len(RB))
print(RB.dim)

import numpy as np
gram_matrix = RB.gramian(fom.h1_0_semi_product)
print(np.max(np.abs(gram_matrix - np.eye(32))))

u = rom.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])
print(u)
U_red = reductor.reconstruct(u)
print(U_red.dim)

ERR = U - U_red
print(ERR.norm(fom.h1_0_semi_product))
fom.visualize((U, U_red, ERR),
              legend=('Detailed', 'Reduced', 'Error'),
              separate_colorbars=True)