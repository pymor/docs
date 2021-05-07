from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')
%matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import pymor.tools.random
pymor.tools.random._default_random_state = None

from pymor.basic import *

problem = StationaryProblem(
      domain=RectDomain(),

      rhs=LincombFunction(
          [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()), ConstantFunction(1., 2)],
          [ProjectionParameterFunctional('mu'), 0.1]),

      diffusion=LincombFunction(
          [ExpressionFunction('1 - x[..., 0]', 2, ()), ExpressionFunction('x[..., 0]', 2, ())],
          [ProjectionParameterFunctional('mu'), 1]),

      dirichlet_data=LincombFunction(
          [ExpressionFunction('2 * x[..., 0]', 2, ()), ConstantFunction(1., 2)],
          [ProjectionParameterFunctional('mu'), 0.5]),

      name='2DProblem'
  )

fom, _ = discretize_stationary_cg(problem, diameter=1/50)

parameter_space = fom.parameters.space((0.1, 1))

training_set = parameter_space.sample_uniformly(100)
validation_set = parameter_space.sample_randomly(20)

from pymor.reductors.neural_network import NeuralNetworkReductor

reductor = NeuralNetworkReductor(fom,
                                 training_set,
                                 validation_set,
                                 l2_err=1e-5,
                                 ann_mse=1e-5)

rom = reductor.reduce(restarts=100)

mu = parameter_space.sample_randomly()

U = fom.solve(mu)
U_red = rom.solve(mu)
U_red_recon = reductor.reconstruct(U_red)

fom.visualize((U, U_red_recon),
              legend=(f'Full solution for parameter {mu}', f'Reduced solution for parameter {mu}'))

test_set = parameter_space.sample_randomly(10)

U = fom.solution_space.empty(reserve=len(test_set))
U_red = fom.solution_space.empty(reserve=len(test_set))

speedups = []

import time

for mu in test_set:
    tic = time.perf_counter()
    U.append(fom.solve(mu))
    time_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    U_red.append(reductor.reconstruct(rom.solve(mu)))
    time_red = time.perf_counter() - tic

    speedups.append(time_fom / time_red)

absolute_errors = (U - U_red).norm()
relative_errors = (U - U_red).norm() / U.norm()

import numpy as np

np.average(absolute_errors)

np.average(relative_errors)

np.median(speedups)

problem = problem.with_(outputs=[('l2', problem.rhs), ('l2_boundary', problem.dirichlet_data)])

fom, _ = discretize_stationary_cg(problem, diameter=1/50)

from pymor.reductors.neural_network import NeuralNetworkOutputReductor

output_reductor = NeuralNetworkOutputReductor(fom,
                                              training_set,
                                              validation_set,
                                              validation_loss=1e-5)

output_rom = output_reductor.reduce(restarts=100)

outputs = []
outputs_red = []
outputs_speedups = []

for mu in test_set:
    tic = time.perf_counter()
    outputs.append(fom.output(mu=mu))
    time_fom = time.perf_counter() - tic

    tic = time.perf_counter()
    outputs_red.append(output_rom.output(mu=mu))
    time_red = time.perf_counter() - tic

    outputs_speedups.append(time_fom / time_red)

outputs = np.squeeze(np.array(outputs))
outputs_red = np.squeeze(np.array(outputs_red))

outputs_absolute_errors = np.abs(outputs - outputs_red)
outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

np.average(outputs_absolute_errors)

np.average(outputs_relative_errors)

np.median(outputs_speedups)