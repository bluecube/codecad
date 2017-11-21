import collections

from .shapes import base
from . import shapes
from . import util

class Part(collections.namedtuple("Part", "data name attributes transform")):
    __slots__ = ()

    def _transformed(self, transform):
        return self.__class__(self.data,
                              self.name,
                              self.attributes,
                              transform * self.transform)

    def _name_prepended(self, name):
        if isinstance(self.name, tuple):
            new_name = (name,) + self.name
        else:
            new_name = (name, self.name)
        return self.__class(self.data,
                            new_name,
                            self.attributes,
                            self.transform)


class Part2D(Part, base.SolidBodyTransformable2D):
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


class Part3D(Part, base.SolidBodyTransformable3D):
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


# Used for storage inside assembly
_PartStorage = collections.namedtuple("_PartStorage", "data attributes transforms")


class FrozenAssembly:
    def __init__(self):
        self._parts = {}
        self._dimension = None

    def dimension(self):
        return self._dimension

    def make_part(self, name, attributes=[]):
        """ Convert this assembly to a part that is insertable into other assemblies. """
        dimension = self.dimension()

        if dimension() is None:
            raise Exception("Making a part of empty assembly is not supported")
        elif dimension == 2:
            return Part2D(self, name, attributes, util.Transformation.zero())
        else:
            return Part3D(self, name, attributes, util.Transformation.zero())

    def all_instances(self):
        """ Returns iterable of all individual parts comprising this assembly,
        recursively descending into subassemblies.

        Names for nested parts are tuples with the inner most name last. This
        format is compatible with assembly's __getitem__ method.

        Transforms for nested parts are merged so that the part can be added to
        the same place with a single step. """

        for name, part_storage in self._parts.items():
            if isinstance(part_storage.data, Assembly):
                for inner_part in part_storage.data.all_instances():
                    renamed_inner_part = inner_part.name_prepended(name)
                    for transform in part_storage.transforms:
                        yield renamed_inner_part._transformed(transform)
            else:
                for transform in part_storage.transforms:
                    yield Part(part_storage.data, name, part_storage.attributes, transform)

    def all_shapes(self):
        """ Returns iterable of all individual shapes comprising this assembly,
        recursively descending into subassemblies. """
        return (part.data for part in self.all_instances())

    def shape(self):
        """ Return single shape for the whole assembly put together """
        return shapes.union(part.data.transformed(part.transform)
                            for part in self.all_instances())

    def __getitem__(self, name):
        """ Find part by name (returns shape or frozen assembly) """
        if isinstance(name, tuple):
            return self._parts[name[0]].data[name[1:]]
        else:
            return self._parts[name].data

    def __iter__(self):
        """ Iterate over part instances of this assembly, not entering subassemblies
        recursively. """
        for name, part_storage in self._parts:
            for transform in part_storage.transforms:
                yield Part(part_storage.data, name, part_storage.attributes, transform)


class Assembly(FrozenAssembly):
    def add(self, part, merge=False):
        """ Add a part into this assembly.
        When adding an assembly, merge argument makes all parts of the added assembly
        become directly included into self.  Default is to not merge.

        Parts are created using `make_part()` method of shape or assembly. """

        dimension = part.data.dimension()
        assert dimension is not None
        if self._dimension is None:
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError("Part has different dimension (2D vs 3D) than the rest of parts in this assembly")

        if merge:
            if not isinstance(part, Assembly):
                raise ValueError("Merging only makes sense when adding subassemblies")

            for inner_part in part:
                self._add_internal(inner_part._transformed(part.transform))
        else:
            self._add_internal(part)

    def _add_internal(self, part):
        if part.name in self._parts:
            existing_part = self._parts[part.name]
            if part.data is not existing_part.data:
                raise ValueError("Name conflict. Part with this name is alredy added and has different data.")
            if part.attributes is not existing_part.attributes:
                raise ValueError("Name conflict. Part with this name is alredy added and has different attributes.")
            existing_part.transforms.append(part.transform)
        else:
            self._parts[part.name] = _PartStorage(part.data, part.attributes, [part.transform])

    def make_part(self, name, attributes=[]):
        """ Convert this assembly to a part that is insertable into other assemblies. """
        frozen = FrozenAssembly()
        frozen._parts = self._parts
        frozen._dimension = self.dimension()

        return frozen.make_part(name, attributes)
