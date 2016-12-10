import itertools
import random
import operator

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


def _contiguous_schedule_recursive(node, ordering_selector, allocator, mapping):
    order = []

    dependencies = list(ordering_selector(node.dependencies))

    # Calculate all dependencies first
    for dep in dependencies:
        if dep in mapping:
            continue # This node has already been processed

        order.extend(_contiguous_schedule_recursive(dep, ordering_selector,
                                                    allocator, mapping))

    # Decref dependency registers before selecting a register for output,
    # because we want to reuse input registers for output when possible
    for dep in dependencies:
        allocator.decref(mapping[dep])

    reg = allocator.allocate(node.refcount)
    mapping[node] = reg

    # Finally output the schedule entry for this node
    order.append(node)

    return order

def _contiguous_schedule(node, ordering_selector):
    allocator = _RegisterAllocator()
    mapping = {}
    order = _contiguous_schedule_recursive(node,
                                           ordering_selector,
                                           allocator,
                                           mapping)

    #print("needs {} registers with selector {}".format(allocator.registers_needed, str(ordering_selector)))

    return allocator.registers_needed, mapping, order

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
