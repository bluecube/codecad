import itertools
import random
import copy

from .node import Node

_last_value = 9999  # Marker for values in schedule passed in lastValue register


class _SchedulerState:
    def __init__(self):
        self._allocations = {}

        # Last non _load/_store node in schedule so far
        # (value of this one is now in lastValue register)
        self.last_node = None

        # Statistics:
        self.registers_needed = 0
        self.store_count = 0
        self.load_count = 0

    def allocate_register(self, refcount):
        for reg in itertools.count():
            if reg not in self._allocations:
                self._allocations[reg] = refcount
                if reg >= self.registers_needed:
                    self.registers_needed = reg + 1
                return reg

    def decref_register(self, reg):
        self._allocations[reg] -= 1
        if not self._allocations[reg]:
            del self._allocations[reg]


def calculate_node_refcounts(node):
    """ Recursively go through all nodes reachable in graph and update their
    reference counts. This must be called before the scheduler is used"""

    assert node.refcount is None
    node.refcount = 1

    for dep in node.dependencies:
        if dep.refcount is None:
            calculate_node_refcounts(dep)
        else:
            dep.refcount += 1


def _contiguous_schedule_recursive(node, need_store, ordering_selector, state):
    assert node.refcount is not None, "Call calculate_node_refcounts() first!"

    if len(node.dependencies) > 2:
        # Breaking up n-ary node to binary.
        # At this point we assume that n-ary (for n > 2) nodes are commutative
        # (because we are splitting the dependencies based on the shufled order)

        dependencies = list(ordering_selector(node.dependencies))

        name = node.name
        params = node.params
        extra_data = node.extra_data

        node.disconnect()

        n = Node(name, params, dependencies[:2], extra_data)
        n.refcount = 1
        for dep in dependencies[2:-1]:
            n = Node(name, params, (n, dep), extra_data)
            n.refcount = 1

        node.connect([dependencies[-1], n])
        dep_order = (1, 0)
    else:
        dep_order = list(
            ordering_selector(list(reversed(range(len(node.dependencies)))))
        )
    # Use reversed dependencies by default, so that the first argument is always
    # evaluated last and we can avoid storing it into a register

    assert 0 <= len(node.dependencies) <= 2

    # Calculate all dependencies first
    for i in dep_order:
        dep = node.dependencies[i]
        if dep.register is None:
            # This node hasn't been processed yet

            dep_need_store = (
                dep.refcount > 1 or i != 0 or dep_order[-1] != 0
            )  # Only last computed dependency can be passed directly

            yield from _contiguous_schedule_recursive(
                dep, dep_need_store, ordering_selector, state
            )

        if i > 0 or state.last_node is not node.dependencies[0]:
            node.dependencies[i] = dep.store_node

    # Decref dependency registers before selecting a register for output,
    # because we want to reuse input registers for output when possible
    for dep in node.dependencies:
        if dep.register != _last_value:
            state.decref_register(dep.register)

    if need_store:
        store_register = state.allocate_register(node.refcount)
    node.register = _last_value

    need_load = (
        len(node.dependencies) > 0
        and node.dependencies[0].name == "_store"
        and state.last_node is not node.dependencies[0].dependencies[0]
    )

    if need_load:
        load_node = Node("_load", (), (node.dependencies[0],))
        load_node.register = _last_value
        load_node.refcount = 1
        node.dependencies[0] = load_node
        state.load_count += 1
        yield load_node

    if len(node.dependencies) > 1:
        state.load_count += 1

    # Finally output the schedule entry for this node
    yield node

    state.last_node = node

    if need_store:
        store_node = Node("_store", (), (node,))
        store_node.register = store_register
        store_node.refcount = node.refcount
        node.refcount = 1
        node.store_node = store_node
        state.store_count += 1
        yield store_node
    else:
        node.store_node = node


def _contiguous_schedule(node, ordering_selector):
    state = _SchedulerState()
    order = list(
        _contiguous_schedule_recursive(
            copy.deepcopy(node), False, ordering_selector, state  # need_store
        )
    )

    # print("needs {} registers with selector {}".format(state.registers_needed, str(ordering_selector)))

    mem_access_count = state.store_count + state.load_count

    return mem_access_count, state.registers_needed, order


def _identity(x):
    return x


def _shuffled(x):
    l = list(x)
    random.shuffle(l)
    return l


def randomized_scheduler(node, random_passes=100):
    return min(
        (
            _contiguous_schedule(node, selector)
            for selector in [_identity, reversed] + [_shuffled] * random_passes
        ),
        key=lambda x: (x[0], x[1]),
    )[1:]
