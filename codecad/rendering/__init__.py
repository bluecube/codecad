import sys
import os
import argparse
import importlib
import re

import flags
import PIL.Image

from .. import assembly


class AssemblyMode(flags.Flags):
    """ Determining how assemblies are treated by the renderer """
    disabled = ()  # Only allow rendering individual shapes, assemblies raise an exception
    whole = ()  # Render the whole assembly at once using `.shape()`
    parts = ()  # Render all assembly parts separaterly using its BoM.
    raw = ()  # Pass the assembly as is


def commandline_render(obj, resolution, default_renderer=None, **kwargs):
    """ Reads commandline arguments, chooses a renderer and passes the parameters to it. """

    parser = argparse.ArgumentParser(description='Render an object')
    parser.add_argument('--output', '-o',
                        help='File name of the output. '
                             'If multiple files are saved, then this has to contain '
                             'exactly one "{}" string which will be replaced by part name.')
    parser.add_argument('--renderer', '-r', choices=_renderers,
                        help='Renderer to use.')
    parser.add_argument('--whole', '-w',
                        action='store_const', const=AssemblyMode.whole, dest='assembly_mode',
                        help='Render the assembly combined shape')
    parser.add_argument('--parts', '-p',
                        action='store_const', const=AssemblyMode.parts, dest='assembly_mode',
                        help='Render all assembly parts from its BoM')

    args = parser.parse_args()

    if args.renderer is not None:
        renderer = args.renderer
    else:
        renderer = default_renderer

    if args.output is not None:
        output = args.output
        if renderer is None:
            renderer = _extensions[os.path.splitext(output)[1]]
    else:
        if renderer is None:
            renderer = "image"
        ext = _renderers[renderer][1]
        if ext is not None:
            output = "output" + ext
        else:
            output = None

    if args.assembly_mode is not None:
        assembly_mode = args.assembly_mode
    else:
        assembly_mode = _renderers[renderer][2]

    if isinstance(obj, assembly.Assembly) and assembly_mode == AssemblyMode.parts:
        pattern = _parse_name_format(output)
        for item in obj.bom():
            _render_one(renderer, item.shape, item.name.join(pattern), resolution, **kwargs)
    else:
        if isinstance(obj, assembly.Assembly):
            if assembly_mode == AssemblyMode.disabled:
                raise ValueError("Renderer {} does not allow assemblies".format(renderer))
            elif assembly_mode == AssemblyMode.whole:
                obj = obj.shape()
        elif isinstance(obj, assembly.Part) or isinstance(obj, assembly.PartTransform):
            obj = obj.shape()

        _render_one(renderer, obj, output, resolution, **kwargs)


def _render_one(renderer, shape, output, resolution, **kwargs):
    if output is not None:
        print("Rendering with renderer {} to file {}".format(renderer, output))
    else:
        print("Rendering with renderer {}".format(renderer))

    _renderers[renderer][0](shape, filename=output, resolution=resolution, **kwargs)


def _parse_name_format(string):
    split = re.split(r'(?<!{){}|{}(?!})', string)
    if len(split) != 2:
        raise ValueError('Filename must contain exactly one occurence of "{}"')
    return split


def _register(name, module_name, extensions, assembly_mode, default_extension=None):
    try:
        module = importlib.import_module("." + module_name, __name__)
    except ImportError as e:
        print("Renderer {} is unavailable due to import error: {}".format(name, str(e)))
        return

    if default_extension is None and len(extensions):
        default_extension = extensions[0]

    for extension in extensions:
        _extensions[extension] = name

    setattr(sys.modules[__name__], module_name, module)
    _renderers[name] = (getattr(module, "render_" + name), default_extension, assembly_mode)


_renderers = {}
_extensions = {}

PIL.Image.init()

_register("image", "image", PIL.Image.EXTENSION.keys(), AssemblyMode.whole, ".png")
_register("stl", "stl_renderer", [".stl"], AssemblyMode.parts)
_register("slice", "matplotlib_slice", [], AssemblyMode.disabled)
_register("mesh", "matplotlib_mesh", [], AssemblyMode.disabled)
_register("gif", "image", [".gif"], AssemblyMode.whole)
_register("svg", "svg", [".svg"], AssemblyMode.disabled)
_register("nodes_graph", "graphviz", [".dot"], AssemblyMode.disabled)
_register("bom", "bom", [".csv"], AssemblyMode.raw)
