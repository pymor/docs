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
import numpy as np

domain = RectDomain(([-1,-1], [1,1]))
indicator_domain = ExpressionFunction(
    '(-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) * (-2/3. <= x[..., 1]) * (x[..., 1] <= -1/3.) * 1. \
   + (-2/3. <= x[..., 0]) * (x[..., 0] <= -1/3.) *  (1/3. <= x[..., 1]) * (x[..., 1] <=  2/3.) * 1.',
    dim_domain=2, shape_range=())
rest_of_domain = ConstantFunction(1, 2) - indicator_domain

f = ExpressionFunction('0.5*pi*pi*cos(0.5*pi*x[..., 0])*cos(0.5*pi*x[..., 1])', dim_domain=2, shape_range=())

parameters = {'diffusion': 2}
thetas = [ExpressionParameterFunctional('1.1 + sin(diffusion[0])*diffusion[1]', parameters,
                                       derivative_expressions={'diffusion': ['cos(diffusion[0])*diffusion[1]',
                                                                             'sin(diffusion[0])']}),
          ExpressionParameterFunctional('1.1 + sin(diffusion[1])', parameters,
                                       derivative_expressions={'diffusion': ['0',
                                                                             'cos(diffusion[1])']}),

                                       ]
diffusion = LincombFunction([rest_of_domain, indicator_domain], thetas)

theta_J = ExpressionParameterFunctional('1 + 1/5 * diffusion[0] + 1/5 * diffusion[1]', parameters,
                                        derivative_expressions={'diffusion': ['1/5','1/5']})

problem = StationaryProblem(domain, f, diffusion, outputs=[('l2', f * theta_J)])

mu_bar = problem.parameters.parse([np.pi/2,np.pi/2])

fom, data = discretize_stationary_cg(problem, diameter=1/50, mu_energy_product=mu_bar)
parameter_space = fom.parameters.space(0, np.pi)

def fom_objective_functional(mu):
    return fom.output(mu)[0]

initial_guess = [0.25, 0.5]

from pymor.discretizers.builtin.cg import InterpolationOperator

diff = InterpolationOperator(data['grid'], problem.diffusion).as_vector(fom.parameters.parse(initial_guess))
fom.visualize(diff)

print(data['grid'])

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.0, 8.0)
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.subplot.bottom'] = .1

from mpl_toolkits.mplot3d import Axes3D # required for 3d plots
from matplotlib import cm # required for colors

import matplotlib.pyplot as plt
from time import perf_counter

def compute_value_matrix(f, x, y):
    f_of_x = np.zeros((len(x), len(y)))
    for ii in range(len(x)):
        for jj in range(len(y)):
            f_of_x[ii][jj] = f((x[ii], y[jj]))
    x, y = np.meshgrid(x, y)
    return x, y, f_of_x

def plot_3d_surface(f, x, y, alpha=1):
    X, Y = x, y
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, f_of_x = compute_value_matrix(f, x, y)
    ax.plot_surface(x, y, f_of_x, cmap='Blues',
                    linewidth=0, antialiased=False, alpha=alpha)
    ax.view_init(elev=27.7597402597, azim=-39.6370967742)
    ax.set_xlim3d([-0.10457963, 3.2961723])
    ax.set_ylim3d([-0.10457963, 3.29617229])
    return ax

def addplot_xy_point_as_bar(ax, x, y, color='orange', z_range=None):
    ax.plot([y, y], [x, x], z_range if z_range else ax.get_zlim(), color)

ranges = parameter_space.ranges['diffusion']
XX = np.linspace(ranges[0] + 0.05, ranges[1], 10)
YY = XX

plot_3d_surface(fom_objective_functional, XX, YY)

reference_minimization_data = {'num_evals': 0,
                               'evaluations' : [],
                               'evaluation_points': [],
                               'time': np.inf}

def record_results(function, data, mu):
    QoI = function(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(fom.parameters.parse(mu).to_numpy())
    data['evaluations'].append(QoI[0])
    return QoI

def report(result, data, reference_mu=None):
    if (result.status != 0):
        print('\n failed!')
    else:
        print('\n succeeded!')
        print('  mu_min:    {}'.format(fom.parameters.parse(result.x)))
        print('  J(mu_min): {}'.format(result.fun[0]))
        if reference_mu is not None:
            print('  absolute error w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(result.x-reference_mu)))
        print('  num iterations:     {}'.format(result.nit))
        print('  num function calls: {}'.format(data['num_evals']))
        print('  time:               {:.5f} seconds'.format(data['time']))
        if 'offline_time' in data:
                print('  offline time:       {:.5f} seconds'.format(data['offline_time']))
        if 'enrichments' in data:
                print('  model enrichments:  {}'.format(data['enrichments']))
    print('')

from functools import partial
from scipy.optimize import minimize

tic = perf_counter()
fom_result = minimize(partial(record_results, fom_objective_functional, reference_minimization_data),
                      initial_guess,
                      method='L-BFGS-B', jac=False,
                      bounds=(ranges, ranges),
                      options={'ftol': 1e-15, 'gtol': 5e-5})
reference_minimization_data['time'] = perf_counter()-tic
reference_mu = fom_result.x

report(fom_result, reference_minimization_data)

reference_plot = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)

for mu in reference_minimization_data['evaluation_points']:
    addplot_xy_point_as_bar(reference_plot, mu[0], mu[1])

from pymor.algorithms.greedy import rb_greedy
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor

coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

training_set = parameter_space.sample_uniformly(25)

RB_reductor = CoerciveRBReductor(fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)

RB_greedy_data = rb_greedy(fom, RB_reductor, training_set, atol=1e-2)

num_RB_greedy_extensions = RB_greedy_data['extensions']
RB_greedy_mus, RB_greedy_errors = RB_greedy_data['max_err_mus'], RB_greedy_data['max_errs']

rom = RB_greedy_data['rom']

print('RB system is of size {}x{}'.format(num_RB_greedy_extensions, num_RB_greedy_extensions))
print('maximum estimated model reduction error over training set: {}'.format(RB_greedy_errors[-1]))

ax = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)

for mu in RB_greedy_mus[:-1]:
    mu = mu.to_numpy()
    addplot_xy_point_as_bar(ax, mu[0], mu[1])

def rom_objective_functional(mu):
    return rom.output(mu)[0]

RB_minimization_data = {'num_evals': 0,
                        'evaluations' : [],
                        'evaluation_points': [],
                        'time': np.inf,
                        'offline_time': RB_greedy_data['time']
                        }

tic = perf_counter()
rom_result = minimize(partial(record_results, rom_objective_functional, RB_minimization_data),
                      initial_guess,
                      method='L-BFGS-B', jac=False,
                      bounds=(ranges, ranges),
                      options={'ftol': 1e-15, 'gtol': 5e-5})
RB_minimization_data['time'] = perf_counter()-tic

report(rom_result, RB_minimization_data, reference_mu)

reference_plot = plot_3d_surface(fom_objective_functional, XX, YY, alpha=0.5)
reference_plot_mean_z_lim = 0.5*(reference_plot.get_zlim()[0] + reference_plot.get_zlim()[1])

for mu in reference_minimization_data['evaluation_points']:
    addplot_xy_point_as_bar(reference_plot, mu[0], mu[1], color='green',
                            z_range=(reference_plot.get_zlim()[0], reference_plot_mean_z_lim))

for mu in RB_minimization_data['evaluation_points']:
    addplot_xy_point_as_bar(reference_plot, mu[0], mu[1], color='orange',
                           z_range=(reference_plot_mean_z_lim, reference_plot.get_zlim()[1]))

def fom_gradient_of_functional(mu):
    return fom.output_d_mu(fom.parameters.parse(mu), return_array=True, use_adjoint=True)

opt_fom_minimization_data = {'num_evals': 0,
                             'evaluations' : [],
                             'evaluation_points': [],
                             'time': np.inf}
tic = perf_counter()
opt_fom_result = minimize(partial(record_results, fom_objective_functional, opt_fom_minimization_data),
                          initial_guess,
                          method='L-BFGS-B',
                          jac=fom_gradient_of_functional,
                          bounds=(ranges, ranges),
                          options={'ftol': 1e-15, 'gtol': 5e-5})
opt_fom_minimization_data['time'] = perf_counter()-tic

# update the reference_mu because this is more accurate!
reference_mu = opt_fom_result.x

report(opt_fom_result, opt_fom_minimization_data)

def rom_gradient_of_functional(mu):
    return rom.output_d_mu(rom.parameters.parse(mu), return_array=True, use_adjoint=True)


opt_rom_minimization_data = {'num_evals': 0,
                             'evaluations' : [],
                             'evaluation_points': [],
                             'time': np.inf,
                             'offline_time': RB_greedy_data['time']}


tic = perf_counter()
opt_rom_result = minimize(partial(record_results, rom_objective_functional, opt_rom_minimization_data),
                  initial_guess,
                  method='L-BFGS-B',
                  jac=rom_gradient_of_functional,
                  bounds=(ranges, ranges),
                  options={'ftol': 1e-15, 'gtol': 5e-5})
opt_rom_minimization_data['time'] = perf_counter()-tic
report(opt_rom_result, opt_rom_minimization_data, reference_mu)

pdeopt_reductor = CoerciveRBReductor(
    fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)

def record_results_and_enrich(function, data, opt_dict, mu):
    U = fom.solve(mu)
    try:
        pdeopt_reductor.extend_basis(U)
        data['enrichments'] += 1
    except:
        print('Extension failed')
    opt_rom = pdeopt_reductor.reduce()
    QoI = opt_rom.output(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(fom.parameters.parse(mu).to_numpy())
    data['evaluations'].append(QoI[0])
    opt_dict['opt_rom'] = rom
    return QoI

def compute_gradient_with_opt_rom(opt_dict, mu):
    opt_rom = opt_dict['opt_rom']
    return opt_rom.output_d_mu(opt_rom.parameters.parse(mu), return_array=True, use_adjoint=True)

opt_along_path_minimization_data = {'num_evals': 0,
                                    'evaluations' : [],
                                    'evaluation_points': [],
                                    'time': np.inf,
                                    'enrichments': 0}
opt_dict = {}
tic = perf_counter()
opt_along_path_result = minimize(partial(record_results_and_enrich, rom_objective_functional,
                                         opt_along_path_minimization_data, opt_dict),
                                 initial_guess,
                                 method='L-BFGS-B',
                                 jac=partial(compute_gradient_with_opt_rom, opt_dict),
                                 bounds=(ranges, ranges),
                                 options={'ftol': 1e-15, 'gtol': 5e-5})
opt_along_path_minimization_data['time'] = perf_counter()-tic

report(opt_along_path_result, opt_along_path_minimization_data, reference_mu)

pdeopt_reductor = CoerciveRBReductor(
    fom, product=fom.energy_product, coercivity_estimator=coercivity_estimator)
opt_rom = pdeopt_reductor.reduce()

def record_results_and_enrich_adaptively(function, data, opt_dict, mu):
    opt_rom = opt_dict['opt_rom']
    primal_estimate = opt_rom.estimate_error(opt_rom.parameters.parse(mu))
    if primal_estimate > 1e-2:
        print('Enriching the space because primal estimate is {} ...'.format(primal_estimate))
        U = fom.solve(mu)
        try:
            pdeopt_reductor.extend_basis(U)
            data['enrichments'] += 1
            opt_rom = pdeopt_reductor.reduce()
        except:
            print('... Extension failed')
    else:
        print('Do NOT enrich the space because primal estimate is {} ...'.format(primal_estimate))
    opt_rom = pdeopt_reductor.reduce()
    QoI = opt_rom.output(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(fom.parameters.parse(mu).to_numpy())
    data['evaluations'].append(QoI[0])
    opt_dict['opt_rom'] = opt_rom
    return QoI

def compute_gradient_with_opt_rom(opt_dict, mu):
    opt_rom = opt_dict['opt_rom']
    return opt_rom.output_d_mu(opt_rom.parameters.parse(mu), return_array=True, use_adjoint=True)

opt_along_path_adaptively_minimization_data = {'num_evals': 0,
                                               'evaluations' : [],
                                               'evaluation_points': [],
                                               'time': np.inf,
                                               'enrichments': 0}
opt_dict = {'opt_rom': opt_rom}
tic = perf_counter()
opt_along_path_adaptively_result = minimize(partial(record_results_and_enrich_adaptively, rom_objective_functional,
                                                    opt_along_path_adaptively_minimization_data, opt_dict),
                                            initial_guess,
                                            method='L-BFGS-B',
                                            jac=partial(compute_gradient_with_opt_rom, opt_dict),
                                            bounds=(ranges, ranges),
                                            options={'ftol': 1e-15, 'gtol': 5e-5})
opt_along_path_adaptively_minimization_data['time'] = perf_counter()-tic

report(opt_along_path_adaptively_result, opt_along_path_adaptively_minimization_data, reference_mu)

print('FOM with finite differences')
report(fom_result, reference_minimization_data, reference_mu)

print('\nROM with finite differences')
report(rom_result, RB_minimization_data, reference_mu)

print('\nFOM with gradient')
report(opt_fom_result, opt_fom_minimization_data, reference_mu)

print('\nROM with gradient')
report(opt_rom_result, opt_rom_minimization_data, reference_mu)

print('\nAlways enrich along the path')
report(opt_along_path_result, opt_along_path_minimization_data, reference_mu)

print('\nAdaptively enrich along the path')
report(opt_along_path_adaptively_result, opt_along_path_adaptively_minimization_data, reference_mu)

assert fom_result.nit == 7
assert opt_along_path_result.nit == 7
assert opt_along_path_minimization_data['num_evals'] == 9
assert opt_along_path_minimization_data['enrichments'] == 9
assert opt_along_path_adaptively_minimization_data['enrichments'] == 4