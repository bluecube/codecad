import inspect
import os
import itertools

from . import nodes

def _(string):
    frame = inspect.stack()[1]
    lines = string.count("\n")
    line = frame[2] - lines

    while string.startswith("\n"):
        string = string[1:]
        line += 1

    return '#line {} "{}"\n'.format(line, frame[1]) + string

def _collect_files():
    root = os.path.dirname(os.path.dirname(__file__))
    sources = []
    for path, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".cl"):
                sources.append(os.path.join(path, filename))

    sources.sort()

    for source in sources:
        yield '#line 1 "{}"'.format(source)
        with open(source, "r") as fp:
            yield fp.read()

def _collect_program_pieces(register_count, node_types_map):
    assert 0 not in node_types_map.values()

    yield _('''
union Word {
    float f;
    struct {
        uchar instruction;
        uchar input1;
        uchar input2;
        uchar output;
    };
};

float4 evaluate(constant Word* program);
''')

    yield from _collect_files()

    yield _('''
float4 evaluate(constant Word* program) {{
    float4 registers[{}];
    while (true) {{
        uchar instruction = program->instruction;
        float4 input1 = registers[program->input1];
        float4 input2 = registers[program->input2];
        float4 *output = &registers[program->output];
        ++program;

        switch (instruction) {{
            case 0:
                return input1;
'''.format(register_count))

    for name, i in node_types_map.items():
        yield _('''
            case {}:
                program += {}(input1, input2, output);
                break;
'''.format(i, name))

    yield _('}}}')

def collect_program():
    return "".join(_collect_program_pieces(32, nodes.Node._type_map))
