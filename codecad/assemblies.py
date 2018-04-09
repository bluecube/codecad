import collections

from .shapes import base
from . import shapes
from . import util


Part = collections.namedtuple("Part", "name data attributes")
Assembly = collections.namedtuple("Assembly", "name instances attributes")


class PartTransformBase(collections.namedtuple("PartTransform", "part transform visible")):
    __slots__ = ()

    def dimension(self):
        return self.part.data.dimension()

    def shape(self):
        """ Return shape corresponding to this part. """
        return self.part.data.transformed(self.transform)

    def _transformed(self, transform):
        return self.__class__(self.part,
                              transform * self.transform,
                              self.visible)

    def hidden(self, hidden=True):
        return self.__class__(self.part,
                              self.transform,
                              not hidden)

    @property
    def name(self):
        return self.part.name

    @property
    def attributes(self):
        return self.part.attributes


class PartTransform2D(PartTransformBase, base.SolidBodyTransformable2D):
    __slots__ = ()

    def translated(self, x, y):
        return self._transformed(util.Transformation.from_degrees((0, 0, 1),
                                                                  0,
                                                                  1,
                                                                  (x, y, 0)))

    def rotated(self, angle):
        return self._transformed(util.Transformation.from_degrees((0, 0, 1),
                                                                  angle,
                                                                  1,
                                                                  (0, 0, 0)))


class PartTransform3D(PartTransformBase, base.SolidBodyTransformable3D):
    __slots__ = ()

    def translated(self, x, y, z):
        return self._transformed(util.Transformation.from_degrees((0, 0, 1),
                                                                  0,
                                                                  1,
                                                                  (x, y, z)))

    def rotated(self, axis, angle):
        return self._transformed(util.Transformation.from_degrees(axis,
                                                                  angle,
                                                                  1,
                                                                  (0, 0, 0)))


class BomItem:
    def __init__(self, name, part):
        self.name = name
        self.part = part
        self.count = 1

    def shape(self):
        """ Return shape corresponding to this bom item. This also makes
        bom items compatible for rendering """
        return self.part.data

    def __str__(self):
        return "{}x {}".format(self.count, self.name)


class AssemblyInterface:
    __slots__ = ()

    def dimension(self):
        return self.part.instances[0].dimension()

    def all_instances(self):
        """ Returns iterable of all individual parts comprising this assembly,
        recursively descending into subassemblies.

        Transforms for nested parts are merged so that the part can be added to
        the same place with a single step. """

        for instance in self:
            if hasattr(instance, "all_instances"):
                for inner_instance in instance.all_instances():
                    yield inner_instance._transformed(instance.transform)
            else:
                yield instance

    def bom(self, recursive=True, visible_only=False):
        """ Iterates over a bill of materials for this assembly, as BomItem instances.
        If recursive is True, goes through all parts in sub assemblies,
        otherwise only lists parts and assemblies directly added to this asm. """
        bom = collections.OrderedDict()

        for instance in (self.all_instances() if recursive else self):
            if visible_only and not instance.visible:
                continue

            name = instance.part.name

            same_names = bom.setdefault(name, [])

            for item in same_names:
                if item.part is instance.part:
                    item.count += 1
                    break
            else:
                name = instance.part.name
                if len(same_names) > 1:
                    name += "-{}".format(len(same_names) + 1)
                same_names.append(BomItem(name, instance.part))

        for same_names in bom.values():
            yield from same_names

    def shape(self):
        """ Return single shape for the whole assembly put together.
        Parts that are not visible are not included. """
        return shapes.union(instance.part.data.transformed(instance.transform)
                            for instance in self.all_instances()
                            if instance.visible).transformed(self.transform)


class AssemblyTransform2D(AssemblyInterface, PartTransform2D):
    __slots__ = ()

    def __iter__(self):
        """ Iterate over part instances of this assembly, not entering subassemblies
        recursively. """
        yield from self.part.instances


class AssemblyTransform3D(AssemblyInterface, PartTransform3D):
    __slots__ = ()

    def __iter__(self):
        """ Iterate over part instances of this assembly, not entering subassemblies
        recursively. """
        yield from self.part.instances


def assembly(name, instances, attributes=[]):
    instances = list(instances)
    if not len(instances):
        raise ValueError("Empty assemblies are not supported")

    dimension = None
    for instance in instances:
        if dimension is None:
            dimension = instance.dimension()
        elif dimension != instance.dimension():
            raise ValueError("Part has different dimension (2D vs 3D) than the rest of parts in this assembly")

    asm = Assembly(name, instances, attributes)

    if dimension == 2:
        return AssemblyTransform2D(asm,
                                   util.Transformation.zero(),
                                   True)
    else:
        return AssemblyTransform3D(asm,
                                   util.Transformation.zero(),
                                   True)
