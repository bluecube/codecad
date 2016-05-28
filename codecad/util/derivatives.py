import theano
from theano import tensor as T
from . import geometry

def gradient(expr, point):
    expr_sum = expr.sum()
    return geometry.Vector(T.grad(expr_sum, point.x, disconnected_inputs="ignore"),
                           T.grad(expr_sum, point.y, disconnected_inputs="ignore"),
                           T.grad(expr_sum, point.z, disconnected_inputs="ignore"))

def fixup_derivatives(expr, point):
    return expr / abs(gradient(expr, point))
