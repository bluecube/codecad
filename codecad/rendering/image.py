import PIL

from . import ray_caster


def render_PIL_image(obj, size=(1024, 768), view_angle=None, resolution=None):
    box = obj.bounding_box()
    box_size = box.size()

    if resolution is None:
        epsilon = min(1, box_size.x, box_size.y, box_size.z) / 10000
    else:
        epsilon = resolution / 10

    camera_params = ray_caster.get_camera_params(box, size, view_angle)

    pixels = ray_caster.render(obj, size=size, epsilon=epsilon, *camera_params)
    return PIL.Image.fromarray(pixels)


def render_image(obj, filename, size=(1024, 768), view_angle=None, resolution=None):
    render_PIL_image(obj, size, view_angle, resolution).save(filename)


def render_gif(obj, filename, size=(640, 480),
               view_angle=None,
               duration=5,
               fps=20,
               loop=True,
               resolution=None):

    with util.status_block("calculating bounding box"):
        box = obj.bounding_box().eval({animation.time: 0})

    if resolution is None:
        epsilon = min(box_size.x, box_size.y, box_size.z) / 10000
    else:
        epsilon = resolution / 10

    camera_params = ray_caster.get_camera_params(box, size, view_angle)

    frame_duration = int(1000 / fps)  # Frame duration in milliseconds
    count = round(1000 * duration / frame_duration)

    frames = []
    for i in range(count):
        time = (i * frame_duration) / 1000
        with util.status_block("rendering frame {}/{}".format(i + 1, count)):
            pixels = ray_caster.render(obj, size=size, epsilon=epsilon, *camera_params)
            frame = PIL.Image.fromarray(pixels)
            frames.append(frame)

    with util.status_block("saving"):
        argv = {}
        if loop:
            argv["loop"] = 0

        frames[0].save(filename,
                       append_images=frames[1:],
                       save_all=True,
                       duration=frame_duration,
                       **argv)
