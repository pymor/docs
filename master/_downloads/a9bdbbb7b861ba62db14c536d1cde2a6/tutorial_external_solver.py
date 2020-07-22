%matplotlib inline

import pymor.tools.random
pymor.tools.random._default_random_state = None

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'pymor.discretizers.builtin.gui.jupyter')

import sys
sys.path.insert(0, 'source/minimal_cpp_demo/build')

import model
mymodel = model.DiffusionOperator(10, 0, 1)
myvector = model.Vector(10, 0)
mymodel.apply(myvector, myvector)
dir(model)

from pymor.operators.interface import Operator
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace

import numpy as np
import math
from model import Vector, DiffusionOperator


class WrappedVector(CopyOnWriteVector):

    def __init__(self, vector):
        assert isinstance(vector, Vector)
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    def to_numpy(self, ensure_copy=False):
        # Note how this uses the buffer protocol setup to allow efficient
        # data access as a Numpy Vector
        result = np.frombuffer(self._impl, dtype=np.float)
        if ensure_copy:
            result = result.copy()
        return result

    def _copy_data(self):
        # This uses the second exposed 'init' signature to delegate to the C++ copy constructor
        self._impl = Vector(self._impl)

    def _scal(self, alpha):
        self._impl.scal(alpha)

    def _axpy(self, alpha, x):
        self._impl.axpy(alpha, x._impl)

    def dot(self, other):
        return self._impl.dot(other._impl)

    def l1_norm(self):
        raise NotImplementedError

    def l2_norm(self):
        return math.sqrt(self.dot(self))

    def l2_norm2(self):
        return self.dot(self)

    def sup_norm(self):
        raise NotImplementedError

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError

class WrappedVectorSpace(ListVectorSpace):

    def __init__(self, dim):
        self.dim = dim

    def zero_vector(self):
        return WrappedVector(Vector(self.dim, 0))

    def make_vector(self, obj):
        # obj is a `model.Vector` instance
        return WrappedVector(obj)

    def __eq__(self, other):
        return type(other) is WrappedVectorSpace and self.dim == other.dim

class WrappedDiffusionOperator(Operator):
    def __init__(self, op):
        assert isinstance(op, DiffusionOperator)
        self.op = op
        self.source = WrappedVectorSpace(op.dim_source)
        self.range = WrappedVectorSpace(op.dim_range)
        self.linear = True

    @classmethod
    def create(cls, n, left, right):
        return cls(DiffusionOperator(n, left, right))

    def apply(self, U, mu=None):
        assert U in self.source

        def apply_one_vector(u):
            v = Vector(self.range.dim, 0)
            self.op.apply(u._impl, v)
            return v

        return self.range.make_array([apply_one_vector(u) for u in U._list])

from pymor.algorithms.pod import pod
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
from pymor.models.basic import InstationaryModel
from pymor.discretizers.builtin import OnedGrid
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.basic import InstationaryRBReductor


def discretize(n, nt, blocks):
    h = 1. / blocks
    ops = [WrappedDiffusionOperator.create(n, h * i, h * (i + 1)) for i in range(blocks)]
    pfs = [ProjectionParameterFunctional('diffusion_coefficients', blocks, i) for i in range(blocks)]
    operator = LincombOperator(ops, pfs)

    initial_data = operator.source.zeros()

    rhs_vec = operator.range.zeros()
    rhs_data = rhs_vec._data[0]
    rhs_data[:] = np.ones(len(rhs_data))
    rhs_data[0] = 0
    rhs_data[len(rhs_data) - 1] = 0
    rhs = VectorOperator(rhs_vec)

    # we can re-use pyMOR's builtin grid and visualizer for our demonstration
    grid = OnedGrid(domain=(0, 1), num_intervals=n)
    visualizer = OnedVisualizer(grid)

    time_stepper = ExplicitEulerTimeStepper(nt)

    fom = InstationaryModel(T=1, operator=operator, rhs=rhs, initial_data=initial_data,
                            time_stepper=time_stepper, num_values=20,
                            visualizer=visualizer, name='C++-Model')
    return fom

%matplotlib inline
# discretize
fom = discretize(50, 10000, 4)
parameter_space = fom.parameters.space(0.1, 1)

# generate solution snapshots
snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(2):
    snapshots.append(fom.solve(mu))

# apply POD
reduced_basis = pod(snapshots, modes=4)[0]

# reduce the model
reductor = InstationaryRBReductor(fom, reduced_basis, check_orthonormality=True)
rom = reductor.reduce()

# stochastic error estimation
mu_max = None
err_max = -1.
for mu in parameter_space.sample_randomly(10):
    U_RB = reductor.reconstruct(rom.solve(mu))
    U = fom.solve(mu)
    err = np.max((U_RB-U).l2_norm())
    if err > err_max:
        err_max = err
        mu_max = mu

# visualize maximum error solution
U_RB = (reductor.reconstruct(rom.solve(mu_max)))
U = fom.solve(mu_max)
fom.visualize((U_RB, U), title=f'mu = {mu}', legend=('reduced', 'detailed'))

fom.visualize((U-U_RB), title=f'mu = {mu}', legend=('error'))