import collections

class NodeCache:
    def __init__(self):
        self._cache = {}

    def make_node(self, name, params, dependencies, extra_data = None):
        node = Node(name, params, dependencies, extra_data)

        #TODO: Maybe try caching subsets of dependencies of >2-ary nodes?
        try:
            cached = self._cache[node]
            assert cached is not Node
            node.disconnect() # Don't let node increase dependencies' refcount
                              # when we're not using it
            return cached
        except KeyError:
            pass
        self._cache[node] = node
        return node

class Node:
    # Mapping of node names to instruction codes
    # These also need to be implemented in opencl.
    _type_map = collections.OrderedDict((name, i + 1) for i, name in enumerate([
        "rectangle", "circle",
        "box", "sphere", "cylinder", "extrusion", "revolution",
        "union", "intersection", "subtraction", "shell",
        "transformation_to", "transformation_from",
        "repetition"]))

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

def get_shape_nodes(shape):
    cache = NodeCache() # TODO: Figure out how to share cache between shapes?
    point = cache.make_node("_point", (), ())
    return shape.get_node(point, cache)
