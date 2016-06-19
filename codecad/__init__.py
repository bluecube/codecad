from .shapes import *
from .rendering import *
from .volume import *
from . import util
from . import unsafe
from . import naca_airfoil

import theano

theano.config.openmp = True
theano.config.openmp_elemwise_minsize = 1000
