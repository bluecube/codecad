import PIL

from . import ray_caster
from . import bitmap


def render_pil_image(obj, size=(1024, 768), view_angle=None):
    if obj.dimension() == 2:
        pixels = bitmap.render(obj, size)
    else:
        camera_params = ray_caster.get_camera_params(
            obj.bounding_box(), size, view_angle
        )
        pixels = ray_caster.render(obj, size=size, *camera_params)

    return PIL.Image.fromarray(pixels)


def render_image(obj, filename, size=(1024, 768), view_angle=None):
    render_pil_image(obj, size, view_angle).save(filename)
