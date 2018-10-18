from .program import make_program, make_program_buffer

from . import node
from . import codegen

codegen.generate_eval_source_code(node.Node, 512)

# pylama:ignore=W0611
