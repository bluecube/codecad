""" Implements nodes of calculation for evaluating the scene.
Also handles generating the OpenCL code for evaluation """

import collections
import itertools

_Variable = object()


class Node:
    # Mapping of node name to tuple (number of parameters, number of input nodes, instruction code)
    # These are also used for code generation and need to be implemented in opencl.
    _node_types = collections.OrderedDict((name, (params, arity, i))
                                          for i, (name, params, arity)
                                          in enumerate([
        # noqa
        # Special nodes:
        ("_return", 0, 1), ("_store", 0, 1), ("_load", 0, 1),

        # Unary nodes:
        # 2D shapes:
        ("rectangle", 2, 1), ("circle", 1, 1),
        ("regular_polygon2d", 2, 1), ("polygon2d", _Variable, 1),
        # 3D shapes:
        ("sphere", 1, 1), ("half_space", 0, 1),
        ("revolution_to", 0, 1), ("twist_revolution_to", 2, 1),
        # Common:
        ("initial_transformation_to", 7, 0),
        ("transformation_to", 7, 1), ("transformation_from", 4, 1),
        ("mirror", 0, 1),
        ("symmetrical_to", 0, 1),
        ("offset", 1, 1), ("shell", 1, 1),
        # Misc:
        ("repetition", 3, 1),
        ("circular_repetition_to", 1, 1), ("circular_repetition_from", 1, 2),
        ("involute_gear", 2, 1),

        # Binary nodes:
        # 3D shapes:
        ("extrusion", 1, 2),
        ("revolution_from", 0, 2), ("twist_revolution_from", 3, 2),
        ("symmetrical_from", 0, 2),
        # Common:
        ("union", 1, 2), ("intersection", 1, 2), ("subtraction", 1, 2),
        ]))

    def __init__(self, name, params, dependencies, extra_data=None):
        # Note: If dependency count > 2, then we assume that the node is  both
        # associative and commutative and that it can be safely broken binary
        # nodes of the same type in any order

        expected_param_count, expected_dependency_count, _ = self._node_types[name]

        self.name = name

        self.params = tuple(params)
        assert len(self.params) == expected_param_count or expected_param_count is _Variable

        self.dependencies = ()
        self.extra_data = extra_data
        self._hash = hash((name, self.params, self.dependencies))

        self.connect(dependencies)
        assert len(self.dependencies) == expected_dependency_count or \
            len(self.dependencies) > expected_dependency_count == 2

        # Values calculated during scheduling:
        self.refcount = None  # How many times is this node referenced by other node
        self.register = None  # Register allocated for output of this node
        self.store_node = None  # Node that stores this node's output value, for visualisation

    def disconnect(self):
        """ Disconnect this node from all its dependencies """
        self.dependencies = ()

    def connect(self, dependencies):
        assert len(self.dependencies) == 0
        self.dependencies = list(dependencies)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.name == other.name and \
               self.params == other.params and \
               self.dependencies == other.dependencies

    def __str__(self):
        return "{}({!r}, {!r}, dependency registers={}, refcount={}, register={})".format(
            self.__class__.__name__,
            self.name,
            self.params,
            [x.register for x in self.dependencies],
            self.refcount,
            self.register)
