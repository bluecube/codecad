import sys
import os
import argparse
import importlib

def commandline_render(shape, resolution, default_renderer="guess"):
    """ Reads commandline arguments, chooses a renderer and passes the parameters to it. """

    parser = argparse.ArgumentParser(description='Render an object')
    parser.add_argument('--output', '-o', default="output.png",
                        help='File name of the output.')
    parser.add_argument('--renderer', '-r', default=default_renderer, choices=_renderers,
                        help='Renderer to use.')
    args = parser.parse_args()

    print("Rendering with renderer {} to file {}".format(args.renderer, args.output))
    _renderers[args.renderer](shape,
                              filename=args.output,
                              resolution=resolution)

def _guess(shape, filename, resolution):
    if filename.lower().endswith(".stl"):
        r = "stl"
    else:
        r = "picture"

    print("Guessed renderer {}".format(r))

    _renderers[r](shape,
                  filename=filename,
                  resolution=resolution)

_modules = {"picture": "ray_caster",
            "stl": "stl_renderer",
            "slice": "matplotlib_slice",
            "gif": "ray_caster"}

_renderers = {"guess": _guess}

for name, module_name in _modules.items():
    try:
        module = importlib.import_module("." + module_name, __name__)
    except ImportError as e:
        print("Renderer {} is unavailable due to import error: {}".format(name, str(e)))
        continue

    setattr(sys.modules[__name__], module_name, module)
    _renderers[name] = getattr(module, "render_" + name)

