from .. import opencl_manager
from . import node


def generate_eval_source_code(node_class, register_count):
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
    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_handler(c, name, params, arity, code)
    c.append('''
       }
    }
}
             ''')


def _generate_op_decl(c_file, name, params, arity, code):
    if params is node._Variable:
        c_file.append('''
uint {}_op(
    __constant float* params,'''.format(name))
    else:
        c_file.append('''
void {}_op('''.format(name))
        for i in range(params):
            c_file.append('''
    float parameter{},'''.format(i + 1))
    assert 1 <= arity <= 2
    for i in range(arity):
        c_file.append('''
    float4 input{},'''.format(i + 1))
    c_file.append('''
    float4* output);''')


def _generate_op_handler(c_file, name, params, arity, code):
    c_file.append('''
            case {}:'''.format(code))
    if params is node._Variable:
        c_file.append('''
                program += {}_op(
                    program,'''.format(name))
    else:
        c_file.append('''
                {}_op('''.format(name))
        for i in range(params):
            c_file.append('''
                    program[{}],'''.format(i))
    assert 1 <= arity <= 2
    for i in range(arity):
        c_file.append('''
                    registers[instruction.input{}],'''.format(i + 1))
    c_file.append('''
                    &registers[instruction.output]);''')
    if params is not node._Variable and params != 0:
        c_file.append('''
                program += {};'''.format(params))
    c_file.append('''
                break;''')
