import collections

from .shapes import base
from . import shapes
from . import util


class Part(collections.namedtuple("Part", "name data attributes")):
    __slots__ = ()

    def shape(self):
        """ Return shape corresponding to this part. """
        return self.data.shape()

    def assembly(self):
        """ Return an assembly containing just theis part. """
        return Assembly([self])


class PartTransform(collections.namedtuple("PartPlacer", "part transform")):
    __slots__ = ()

    def shape(self):
        """ Return shape corresponding to this part. """
        return self.part.shape().transformed(self.transform)

    def assembly(self):
        """ Return an assembly containing just theis part. """
        return Assembly([self])

    def _transformed(self, transform):
        return self.__class__(self.part,
                              transform * self.transform)


class PartTransform2D(PartTransform, base.SolidBodyTransformable2D):
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


class PartTransform3D(PartTransform, base.SolidBodyTransformable3D):
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


class _FrozenAssembly:
    def __init__(self, instances, dimension):
        self._instances = instances
        self._dimension = dimension

    def dimension(self):
        return self._dimension

    def make_part(self, name, attributes=[]):
        """ Convert this assembly to a part that is insertable into other assemblies. """
        dimension = self.dimension()

        if dimension is None:
            raise Exception("Making a part of empty assembly is not supported")
        elif dimension == 2:
            return PartTransform2D(Part(name, self, attributes), util.Transformation.zero())
        else:
            return PartTransform3D(Part(name, self, attributes), util.Transformation.zero())

    def all_instances(self):
        """ Returns iterable of all individual parts comprising this assembly,
        recursively descending into subassemblies.

        Transforms for nested parts are merged so that the part can be added to
        the same place with a single step. """

        for instance in self._instances:
            if hasattr(instance.part.data, "all_instances"):
                for inner_instance in instance.part.data.all_instances():
                    yield inner_instance._transformed(instance.transform)
            else:
                yield instance

    def bom(self):
        """ Returns a bill of materials for this assembly Bill of materials is
        an iterable of BomItem records. """
        bom = collections.OrderedDict()

        for instance in self.all_instances():
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
        """ Return single shape for the whole assembly put together """
        return shapes.union(instance.part.data.transformed(instance.transform)
                            for instance in self.all_instances())

    def assembly(self):
        """ Return self for compatibility with shapes and parts """
        return self

    def __iter__(self):
        """ Iterate over part instances of this assembly, not entering subassemblies
        recursively. """
        return iter(self._parts)


class Assembly(_FrozenAssembly):
    def __init__(self, instances=[]):
        super().__init__([], None)
        for instance in instances:
            self.add(instance)

    def add(self, instance):
        """ Add a part instance into this assembly.
        Parts are created using `make_part()` method of shapes and assemblies. """

        dimension = instance.part.data.dimension()
        assert dimension is not None
        if self._dimension is None:
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError("Part has different dimension (2D vs 3D) than the rest of parts in this assembly")

        self._instances.append(instance)

    def make_part(self, name, attributes=[]):
        """ Convert this assembly to a part that is insertable into other assemblies. """
        return _FrozenAssembly(self._instances, self._dimension).make_part(name, attributes)
