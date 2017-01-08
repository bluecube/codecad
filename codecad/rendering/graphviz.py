from ..compute import program, scheduler

def render_nodes_graph(shape, resolution, filename):
    shape_node = program.get_shape_nodes(shape)

    _, ordered = scheduler.randomized_scheduler(shape_node)

    with open(filename, "w") as fp:
        fp.write("digraph Nodes {\n");
        fp.write("  graph[concentrate=true];\n");
        stack = [shape_node]

        for i, node in enumerate(ordered):
            fp.write('  node{} [label="{}\\n{}({})\\nrefcount {}\\n->r{}"];\n'.format(id(node),
                                                                                      i,
                                                                                      node.name,
                                                                                      ", ".join(str(p) for p in node.params),
                                                                                      node.refcount,
                                                                                      node.register))
            for dep in node.dependencies:
                fp.write('  node{} -> node{};\n'.format(id(dep), id(node)))

        fp.write("}")
