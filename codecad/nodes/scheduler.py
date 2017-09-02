import itertools
import random
import operator
import copy

from . import node

class _RegisterAllocator:
    def __init__(self):
        self._allocations = {}
        self.registers_needed = 0

    def allocate(self, refcount):
        for reg in itertools.count():
            if reg not in self._allocations:
                self._allocations[reg] = refcount
                if reg >= self.registers_needed:
                    self.registers_needed = reg + 1
                return reg

    def decref(self, reg):
        self._allocations[reg] -= 1
        if not self._allocations[reg]:
            del self._allocations[reg]


def _contiguous_schedule_recursive(node, ordering_selector, allocator):
    order = []

    dependencies = list(ordering_selector(node.dependencies))

    # Breaking up n-ary node to binary
    if len(dependencies) > 2:
        name = node.name
        params = node.params
        extra_data = node.extra_data

        node.disconnect()

        n = nodes.Node(name, params, dependencies[:2], extra_data)
        for dep in dependencies[2:-1]:
            n = nodes.Node(name, params, (n, dep), extra_data)

        dependencies = (n, dependencies[-1])

        node.connect(dependencies)


    # Calculate all dependencies first
    for dep in dependencies:
        if dep.register is not None:
            continue # This node has already been processed

        order.extend(_contiguous_schedule_recursive(dep, ordering_selector, allocator))

    # Decref dependency registers before selecting a register for output,
    # because we want to reuse input registers for output when possible
    for dep in dependencies:
        allocator.decref(dep.register)

    node.register = allocator.allocate(node.refcount)

    # Finally output the schedule entry for this node
    order.append(node)

    return order

def _contiguous_schedule(node, ordering_selector):
    allocator = _RegisterAllocator()
    order = _contiguous_schedule_recursive(copy.deepcopy(node),
                                           ordering_selector,
                                           allocator)

    #print("needs {} registers with selector {}".format(allocator.registers_needed, str(ordering_selector)))

    return allocator.registers_needed, order

def _identity(x):
    return x

def _shuffled(x):
    l = list(x)
    random.shuffle(l)
    return l

def randomized_scheduler(node, random_passes = 100):
    return min((_contiguous_schedule(node, selector)
                for selector in
                [_identity, reversed] + [_shuffled] * random_passes),
               key=operator.itemgetter(0))
