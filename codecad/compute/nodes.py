import collections

class Node:
    # Mapping of node names to instruction codes
    # These also need to be implemented in opencl.
    _type_map = collections.OrderedDict((name, i + 2) for i, name in enumerate([
        "rectangle", "circle", # 2D shapes
        "sphere", "extrusion", "revolution", # 3D shapes
        "union", "intersection", "subtraction", "shell", # Common operations
        "transformation_to", "transformation_from", # Transform
        "repetition"])) # Misc

    def __init__(self, name, params, dependencies, extra_data = None):
        # Note: If dependency count > 2, then we assume that the node is  both
        # associative and commutative and that it can be safely broken binary
        # nodes of the same type in any order
        assert name in self._type_map or name.startswith("_")
        self.name = name
        self.params = tuple(params)
        self.dependencies = ()
        self.extra_data = extra_data
        self._hash = hash((name, self.params, self.dependencies))

        self.refcount = 0 # How many times is this node referenced by other node
        self.connect(dependencies)

        self.register = None # Register allocated for output of this node

    def disconnect(self):
        for dep in self.dependencies:
            dep.refcount -= 1
        self.dependencies = ()

    def connect(self, dependencies):
        assert len(self.dependencies) == 0
        self.dependencies = tuple(dependencies)
        for dep in self.dependencies:
            dep.refcount += 1

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.name == other.name and \
               self.params == other.params and \
               self.dependencies == other.dependencies
