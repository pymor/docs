%matplotlib inline

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')

from pymor.basic import *

domain = RectDomain([[0.,0.], [1.,1.]])

diffusion = ConstantFunction(1, 2)

rhs = ExpressionFunction('(sqrt( (x[...,0]-0.5)**2 + (x[...,1]-0.5)**2) <= 0.3) * 1.', 2, ())

problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    rhs=rhs,
)

m, data = discretize_stationary_cg(problem, diameter=1/4)

U = m.solve()

m.visualize(U)

m, data = discretize_stationary_cg(problem, diameter=1/4, grid_type=RectGrid)
m.visualize(m.solve())

m, data = discretize_stationary_fv(problem, diameter=1/4, grid_type=TriaGrid)
m.visualize(m.solve())

set_log_levels({'pymor': 'WARN'})

domain = RectDomain(bottom='neumann')

neumann_data = ConstantFunction(-1., 2)

diffusion = ExpressionFunction('1. - (sqrt( (x[...,0]-0.5)**2 + (x[...,1]-0.5)**2) <= 0.3) * 0.999' , 2, ())

problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=neumann_data
)

m, data = discretize_stationary_cg(problem, diameter=1/32)
m.visualize(m.solve())

diffusion = ExpressionFunction(
    '1. - (sqrt( (np.mod(x[...,0],1./K)-0.5/K)**2 + (np.mod(x[...,1],1./K)-0.5/K)**2) <= 0.3/K) * 0.999',
    2, (),
    values={'K': 10}
)

problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=neumann_data
)


m, data = discretize_stationary_cg(problem, diameter=1/100)
m.visualize(m.solve())

diffusion = BitmapFunction('RB.png', range=[0.001, 1])
problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=neumann_data
)

m, data = discretize_stationary_cg(problem, diameter=1/100)
m.visualize(m.solve())

neumann_data = ExpressionFunction('-cos(pi*x[...,0])**2*neum[0]', 2, (), parameters= {'neum': 1})

diffusion = ExpressionFunction(
    '1. - (sqrt( (np.mod(x[...,0],1./K)-0.5/K)**2 + (np.mod(x[...,1],1./K)-0.5/K)**2) <= 0.3/K) * 0.999',
    2, (),
    values={'K': 10}
)
problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=neumann_data
)

m, data = discretize_stationary_cg(problem, diameter=1/100)
m.parameters

m.visualize(m.solve({'neum': [1.]}))

m.visualize(m.solve(-100))

diffusion = ExpressionFunction(
    '1. - (sqrt( (np.mod(x[...,0],1./K)-0.5/K)**2 + (np.mod(x[...,1],1./K)-0.5/K)**2) <= 0.3/K) * (1 - diffu[0])',
    2, (),
    values={'K': 10},
    parameters= {'diffu': 1}
)

problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=neumann_data
)

m, data = discretize_stationary_cg(problem, diameter=1/100)
m.parameters

m.visualize(m.solve({'diffu': 0.001, 'neum': 1}))

m.visualize(m.solve([1, -1]))

f_R = BitmapFunction('R.png', range=[1, 0])
f_B = BitmapFunction('B.png', range=[1, 0])

theta_R = ExpressionParameterFunctional('R[0] - 1', {'R': 1})
theta_B = ExpressionParameterFunctional('B[0] - 1', {'B': 1})

diffusion = LincombFunction(
    [ConstantFunction(1., 2), f_R, f_B],
    [1., theta_R, theta_B]
)
diffusion.parameters

problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion,
    neumann_data=ConstantFunction(-1, 2)
)
m, data = discretize_stationary_cg(problem, diameter=1/100)
m.visualize((m.solve([1., 0.001]), m.solve([0.001, 1])))

m.operator