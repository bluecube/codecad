from .. import opencl_manager
from . import node


def generate_eval_source_code(node_class, register_count):
    opencl_manager.instance.max_register_count = register_count

    h = opencl_manager.instance.common_header
    h.append('float4 evaluate(__constant float* program, float3 point);')

    c = opencl_manager.instance.add_compile_unit()
    c.append('#define EVAL_REGISTER_COUNT {}'.format(register_count))
    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_decl(c, name, params, arity, code)
    c.append('''
float4 evaluate(__constant float* program, float3 point)
{
    float4 registers[EVAL_REGISTER_COUNT];

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
        {''')
    c.append('''
            case {}:
                return registers[instruction.input1];'''.format(node_class._node_types["_return"][2]))
    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_handler(c, name, params, arity, code)
    c.append('''
       }
    }
}
             ''')


def _generate_op_decl(c_file, name, params, arity, code):
    if name[0] == "_":
        return  # Underscore nodes have hard coded implementation

    c_file.append('''
float4 {}_op('''.format(name))
    if params is node._Variable:
        c_file.append('''
    __constant float* restrict* restrict params,''')
    else:
        for i in range(params):
            c_file.append('''
    float parameter{},'''.format(i + 1))
    if arity == 0:
        c_file.append('''
    float3 point);''')
    else:
        assert 1 <= arity <= 2
        for i in range(arity - 1):
            c_file.append('''
    float4 input{},'''.format(i + 1))
        c_file.append('''
    float4 input{});'''.format(arity))


def _generate_op_handler(c_file, name, params, arity, code):
    if name[0] == "_":
        return  # Underscore nodes don't need declaration

    c_file.append('''
            case {}:
                registers[instruction.output] = {}_op('''.format(code, name))
    if params is node._Variable:
        c_file.append('''
                    &program,'''.format(name))
    else:
        for i in range(params):
            c_file.append('''
                    program[{}],'''.format(i))
    if arity == 0:
        c_file.append('''
                    point);''')
    else:
        assert 1 <= arity <= 2
        for i in range(arity - 1):
            c_file.append('''
                    registers[instruction.input{}],'''.format(i + 1))
        c_file.append('''
                    registers[instruction.input{}]);'''.format(arity))
    if params is not node._Variable and params != 0:
        c_file.append('''
                program += {};'''.format(params))
    c_file.append('''
                break;''')
