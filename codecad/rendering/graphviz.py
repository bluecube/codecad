from ..compute import nodes

def render_nodes_graph(shape, resolution, filename):
    shape_node = nodes.get_shape_nodes(shape)

    visited_ids = set()

    with open(filename, "w") as fp:
        fp.write("digraph Nodes {\n");
        stack = [shape_node]

        while len(stack):
            node = stack.pop()

            if id(node) in visited_ids:
                continue
            else:
                visited_ids.add(id(node))

            fp.write('  node{} [label="{}({})"];\n'.format(id(node),
                                                           node.name,
                                                           ", ".join(str(p) for p in node.params)))
            for dep in node.dependencies:
                stack.append(dep)
                fp.write('  node{} -> node{};\n'.format(id(dep), id(node)))
        fp.write("}")
