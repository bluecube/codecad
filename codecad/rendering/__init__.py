import sys
import os
import argparse
import importlib

import PIL.Image

def commandline_render(shape, resolution, default_renderer=None):
    """ Reads commandline arguments, chooses a renderer and passes the parameters to it. """

    parser = argparse.ArgumentParser(description='Render an object')
    parser.add_argument('--output', '-o',
                        help='File name of the output.')
    parser.add_argument('--renderer', '-r', choices=_renderers,
                        help='Renderer to use.')
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
            renderer = "picture"
        ext = _renderers[renderer][1]
        if ext is not None:
            output = "output" + ext

    if output is not None:
        print("Rendering with renderer {} to file {}".format(renderer, output))
    else:
        print("Rendering with renderer {}".format(renderer))

    _renderers[renderer][0](shape, filename=output, resolution=resolution)

def _register(name, module_name, extensions, default_extension = None):
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
    _renderers[name] = (getattr(module, "render_" + name), default_extension)

_renderers = {}
_extensions = {}

PIL.Image.init()

_register("picture", "ray_caster", PIL.Image.EXTENSION.keys(), ".png")
_register("stl", "stl_renderer", [".stl"])
_register("slice", "matplotlib_slice", [])
_register("gif", "ray_caster", [".gif"])
_register("svg", "svg", [".svgz", ".svg"])
