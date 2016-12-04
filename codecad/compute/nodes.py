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

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.name == other.name and \
               self.params == other.params and \
               self.dependencies == other.dependencies
