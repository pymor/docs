%matplotlib inline

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')

from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg

p = thermal_block_problem((2,2))
fom, _ = discretize_stationary_cg(p, diameter=1/100)

U = fom.solve([1., 0.1, 0.1, 1.])
fom.visualize(U)

from pymor.algorithms.pod import pod
from matplotlib import pyplot as plt

snapshots = fom.solution_space.empty()
for mu in p.parameter_space.sample_randomly(20):
    snapshots.append(fom.solve(mu))
basis, singular_values = pod(snapshots, modes=10)

_ = plt.semilogy(singular_values)

from pymor.tools.formatsrc import print_source
print_source(fom.solve)

print_source(fom._solve)

type(fom)

fom.rhs

fom.rhs.source

from pymor.operators.interface import Operator
print_source(Operator.as_range_array)

U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=[1., 0.1, 0.1, 1.])

mu = fom.parameters.parse([1., 0.1, 0.1, 1.])
U2 = fom.operator.apply_inverse(fom.rhs.as_range_array(mu), mu=mu)

(U-U2).norm()

reduced_operator = fom.operator.apply2(basis, basis, mu=mu)
reduced_rhs = basis.inner(fom.rhs.as_range_array(mu))

import numpy as np

u_N = np.linalg.solve(reduced_operator, reduced_rhs)
u_N

U_N = basis.lincomb(u_N.T)
U_N

(U-U_N).norm(fom.h1_0_product) / U.norm(fom.h1_0_product)

fom.visualize((U, U_N, U-U_N), separate_colorbars=True)

type(reduced_operator)

from pymor.operators.numpy import NumpyMatrixOperator

reduced_operator = NumpyMatrixOperator(reduced_operator)
reduced_rhs = NumpyMatrixOperator(reduced_rhs)

from pymor.models.basic import StationaryModel
rom = StationaryModel(reduced_operator, reduced_rhs)
rom

u_N2 = rom.solve()
u_N.T - u_N2.to_numpy()

print(fom.parameters)
print(rom.parameters)

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

fom.operator

reduced_operators = [NumpyMatrixOperator(op.apply2(basis, basis))
                     for op in fom.operator.operators]

reduced_operator = fom.operator.with_(operators=reduced_operators)

fom.rhs.parameters

rom = StationaryModel(reduced_operator, reduced_rhs)

rom.parameters

u_N3 = rom.solve(mu)
u_N.T - u_N3.to_numpy()

tic = perf_counter()
fom.solve(mu)
toc = perf_counter()
rom.solve(mu)
tac = perf_counter()
print(f'FOM: {toc-tic:.5f} (s)')
print(f'ROM: {tac-toc:.5f} (s)')

from pymor.algorithms.projection import project

reduced_operator = project(fom.operator, basis, basis)
reduced_rhs      = project(fom.rhs,      basis, None )

reduced_operator

rom = StationaryModel(reduced_operator, reduced_rhs)
u_N4 = rom.solve(mu)
u_N.T - u_N4.to_numpy()

print_source(project)

from pymor.algorithms.projection import ProjectRules
ProjectRules

assert ProjectRules.rules[8].action_description == 'LincombOperator'

ProjectRules.rules[8]

assert ProjectRules.rules[3].action_description == 'apply_basis'

ProjectRules.rules[3]

from pymor.reductors.basic import StationaryRBReductor

reductor = StationaryRBReductor(fom, basis)
rom = reductor.reduce()

u_N5 = rom.solve(mu)
u_N.T - u_N5.to_numpy()

print_source(reductor.project_operators)

print_source(reductor.build_rom)

U_N5 = reductor.reconstruct(u_N5)
(U_N - U_N5).norm()

print_source(reductor.reconstruct)