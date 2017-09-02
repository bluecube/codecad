""" Implements nodes of calculation for evaluating the scene.
Also handles generating the OpenCL code for evaluation """

import collections

from .. import opencl_manager

class Node:
    # Mapping of node names to instruction codes
    # These also need to be implemented in opencl.
    _type_map = collections.OrderedDict((name, i + 2) for i, name in enumerate([
        "rectangle", "circle", "polygon2d", # 2D shapes
        "sphere", "extrusion", "revolution_to", "revolution_from", # 3D shapes
        "union", "intersection", "subtraction", "offset", "shell", # Common operations
        "transformation_to", "transformation_from", # Transform
        "repetition", "half_space", "involute_gear"])) # Misc

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

    @classmethod
    def generate_eval_source_code(cls, register_count):
        h = opencl_manager.instance.common_header
        h.append('float4 evaluate(__constant float* program, float3 point);')

        c = opencl_manager.instance.add_compile_unit()
        c.append('#define EVAL_REGISTER_COUNT {}'.format(register_count))
        for name in cls._type_map.keys():
            c.append('uchar {}_op(__constant float* params, float4* output, float4 param1, float4 param2);'.format(name))
        c.append('''
float4 evaluate(__constant float* program, float3 point)
{
    float4 registers[EVAL_REGISTER_COUNT];
    registers[0] = as_float4(point);

    while (true)
    {
        union
        {
            float f;
            struct
            {
                uchar opcode;
                uchar output;
                uchar input1;
                uchar input2;
            };
        } instruction;

        instruction.f = *program++;

        switch (instruction.opcode)
        {
            case 0:
                return registers[0];
                 ''')
        for name, i in cls._type_map.items():
            c.append('''
            case {}:
                program += {}_op(program,
                    &registers[instruction.output],
                    registers[instruction.input1],
                    registers[instruction.input2]);
                break;
                     '''.format(i, name))
        c.append('''
       }
    }
}
                 ''')

Node.generate_eval_source_code(32)
