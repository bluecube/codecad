class NodeCache:
    def __init__(self):
        self._cache = {}

    def make_node(self, name, params, dependencies, extra_data = None):
        node = Node(name, params, dependencies, extra_data)
        try:
            return self._cache[node]
        except KeyError:
            pass
        self._cache[node] = node
        return node

class Node:
    def __init__(self, name, params, dependencies, extra_data = None):
        self.name = name
        self.params = tuple(params)
        self.dependencies = tuple(dependencies)
        self.extra_data = extra_data
        self._hash = hash((name, self.params, self.dependencies))

        self.refcount = 0 # How many times is this node referenced by other node
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
    point = cache.make_node("point", (), ())
    return shape.get_node(point, cache)
