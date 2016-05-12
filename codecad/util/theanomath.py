import sys
import functools
import theano
import theano.tensor as T
import numpy

class _TheanoMath:
    @staticmethod
    def is_theano(*args, **kwargs):
        return any(isinstance(x, theano.Variable) for x in args) or \
               any(isinstance(x, theano.Variable) for x in kwargs.values())

    @classmethod
    def maximum(cls, *args):
        if cls.is_theano(*args):
            return functools.reduce(T.maximum, args)
        else:
            return functools.reduce(numpy.maximum, args)

    @classmethod
    def minimum(cls, *args):
        if cls.is_theano(*args):
            return functools.reduce(T.minimum, args)
        else:
            return functools.reduce(numpy.minimum, args)

    def __getattr__(self, name):
        def f(*args, **kwargs):
            if self.is_theano(*args, **kwargs):
                return getattr(T, name)(*args, **kwargs)
            else:
                return getattr(numpy, name)(*args, **kwargs)
        return f

# A trick to allow __getattr__ on a module.
# http://stackoverflow.com/a/7668273/89480
sys.modules[__name__] = _TheanoMath()
