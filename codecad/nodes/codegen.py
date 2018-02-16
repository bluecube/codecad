from ..cl_util import opencl_manager
from . import node, program


def generate_eval_source_code(node_class, register_count):
    opencl_manager.max_register_count = register_count

    h = opencl_manager.common_header
    h.append('float4 evaluate(__constant float* program, float3 point);')

    c = opencl_manager.add_compile_unit()
    c.append_define("EVAL_REGISTER_COUNT", register_count)
    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_decl(c, name, params, arity, code)
    c.append('''
float4 evaluate(__constant float* program, float3 point)
{
    float4 registers[EVAL_REGISTER_COUNT];
    float4 lastValue;

    while (true)
    {
        uint instruction = (uint)(*program++);
        uint opcode = instruction / EVAL_REGISTER_COUNT;
        uint secondaryRegister = instruction % EVAL_REGISTER_COUNT;

        switch (opcode)
        {''')
    c.append('''
            case {}:
                // _return
                return lastValue;'''.format(node_class._node_types["_return"][2]))
    c.append('''
            case {}:
                // _store
                registers[secondaryRegister] = lastValue;
                break;'''.format(node_class._node_types["_store"][2]))
    c.append('''
            case {}:
                // _load
                lastValue = registers[secondaryRegister];
                break;'''.format(node_class._node_types["_load"][2]))
    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_handler(c, name, params, arity, code)
    c.append('''
       }
    }
}''')


def _format_function(before, args):
    return before + "(" + ", ".join(args) + ");"


def _generate_op_decl(c_file, name, params, arity, code):
    if name[0] == "_":
        return  # Underscore nodes don't need declaration

    args = []

    if params is node._Variable:
        args.append('__constant float* restrict* restrict params')
    else:
        args.extend('float param{}'.format(i + 1) for i in range(params))

    if arity == 0:
        args.append('float3 point')
    else:
        assert 1 <= arity <= 2
        args.extend('float4 input{}'.format(i + 1) for i in range(arity))

    c_file.append(_format_function('float4 {}_op'.format(name), args))


def _generate_op_handler(c_file, name, params, arity, code):
    if name[0] == "_":
        return  # Underscore nodes have hard coded implementation

    args = []
    if params is node._Variable:
        args.append("&program")
    else:
        args.extend("program[{}]".format(i) for i in range(params))

    if arity == 0:
        args.append("point")
    else:
        assert 1 <= arity <= 2
        args.append("lastValue")
        if arity == 2:
            args.append("registers[secondaryRegister]")

    c_file.append('''
            case {}:'''.format(code))
    c_file.append(_format_function('''
                lastValue = {}_op'''.format(name), args))
    if params is not node._Variable and params != 0:
        c_file.append('''
                program += {};'''.format(params))
    c_file.append('''
                break;''')


def generate_fixed_eval_source_code(schedule, node_class):
    c = opencl_manager.add_compile_unit()
    c.include_origin = False

    for name, (params, arity, code) in node_class._node_types.items():
        _generate_op_decl(c, name, params, arity, code)
    c.append('''
float4 evaluate(__constant float*, float3 point)
{''')
    registers_declared = set()
    for node in schedule:
        registers_declared.add(node.register)

    for register in sorted(registers_declared):
        c.append('''
    float4 r{};'''.format(register))
    c.append('')

    for node in schedule:
        if node.name == "_return":
            c.append('''
    return r{};'''.format(node.dependencies[0].register))
        elif node.name == "_store" or node.name == "_load":
            c.append('''
    r{} = r{}; // {}'''.format(node.register, node.dependencies[0].register, node.name))
        else:
            args = [str(param) for param in node.params]
            if len(node.dependencies) == 0:
                args.append("point");
            else:
                assert 1 <= len(node.dependencies) <= 2
                args.extend("r{}".format(dep.register) for dep in node.dependencies)

            c.append(_format_function('''
    r{} = {}_op'''.format(node.register, node.name), args))
        c.append('''
        // opcode: {}, secondary register: {}'''.format(*program.get_opcode(node)))
    c.append('''
}''')

    return c
