from ..nodes import program, scheduler, codegen, node


def render_nodes_graph(shape, filename):
    shape_node = program.get_shape_nodes(shape)
    scheduler.calculate_node_refcounts(shape_node)
    _, ordered = scheduler.randomized_scheduler(shape_node)

    with open(filename, "w") as fp:
        fp.write("digraph Nodes {\n")
        fp.write("  graph[concentrate=true];\n")

        for i, n in enumerate(ordered):
            if n.name in ("_store", "_load"):
                fp.write(
                    '  node{} [label="{}\\n{}\\nrefcount {}\\n->r{}",style="filled"];\n'.format(
                        id(n), i, n.name, n.refcount, n.register
                    )
                )
            else:
                fp.write(
                    '  node{} [label="{}\\n{}({})"];\n'.format(
                        id(n), i, n.name, ", ".join(str(p) for p in n.params)
                    )
                )
            for dep, direction in zip(n.dependencies, ["nw", "ne"]):
                fp.write("  node{}:s -> node{}:{};\n".format(id(dep), id(n), direction))

        fp.write("}")


def render_c_evaluator(shape, filename):
    shape_node = program.get_shape_nodes(shape)
    scheduler.calculate_node_refcounts(shape_node)
    _, ordered = scheduler.randomized_scheduler(shape_node)

    c_file = codegen.generate_fixed_eval_source_code(ordered, node.Node)

    with open(filename, "w") as fp:
        fp.write(c_file.code())
