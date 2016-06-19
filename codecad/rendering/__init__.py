import sys
import os
import argparse

from .ray_caster import render_picture
from .stl_renderer import render_stl

def commandline_render(shape, resolution):
    """ Reads commandline arguments, chooses a renderer and passes the parameters to it. """

    parser = argparse.ArgumentParser(description='Render an object')
    parser.add_argument('--output', '-o', default="output.png",
                        help='File name of the output.')
    parser.add_argument('--renderer', '-r', default="guess", choices=_renderers,
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

_renderers = {"guess": _guess,
             "picture": render_picture,
             "stl": render_stl}

