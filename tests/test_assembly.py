import pytest

import codecad

import data
import tools


def test_shape_to_assembly():
    shape = codecad.shapes.sphere()
    part = shape.make_part("sphere")
    asm = codecad.assembly("test_asm", [part])

    instances = list(asm.all_instances())
    assert len(instances) == 1
    assert instances[0].part.data is shape


def test_shape_to_assembly_no_part():
    with pytest.raises(Exception):
        codecad.Assembly("X", [codecad.shapes.sphere()])


@pytest.fixture(scope="module")
def asm_data():
    """ provides a tuple of assembly and corresponding shape (generated explicitly) """
    n = 3
    y_spacing = 2.5

    s = data.bin_counter(0).extruded(0.1)
    row_parts = [s.make_part("shape_0").translated_x(0.5)]
    row_shapes = [s.translated_x(0.5)]

    for i in range(1, n + 1):
        s = data.bin_counter(i).extruded(0.1)

        row_parts.append(codecad.assembly("row_{}".format(i),
                                          [row_parts[-1],
                                           s.make_part("shape_{}".format(i))
                                            .translated_x(0.5 + 2 * i)
                                           ]))
        row_shapes.append(row_shapes[-1] + s.translated_x(0.5 + 2 * i))

    assembly = codecad.assembly("test_assembly", [row.translated_y(i * y_spacing)
                                 for i, row in enumerate(row_parts)])
    shape = codecad.shapes.union([row.translated_y(i * y_spacing)
                                  for i, row in enumerate(row_shapes)])

    return n, assembly, shape


def test_subassemblies_recursive_bom(asm_data):
    n, asm, shape = asm_data
    seen = {}

    for item in asm.bom():
        assert item.name not in seen
        seen[item.name] = item

    for i in range(n + 1):
        name = "shape_{}".format(i)
        assert name in seen
        assert seen[name].count == n - i + 1
        del seen[name]

    assert len(seen) == 0


def test_subassemblies_flat_bom(asm_data):
    n, asm, shape = asm_data
    seen = {}

    for item in asm.bom(recursive=False):
        assert item.name not in seen
        print(item)
        seen[item.name] = item
        assert item.count == 1

    assert "shape_0" in seen
    del seen["shape_0"]

    for i in range(1, n + 1):
        name = "row_{}".format(i)
        assert name in seen
        del seen[name]

    assert len(seen) == 0


def test_subassemblies_shape(asm_data):
    n, asm, shape = asm_data
    tools.assert_shapes_equal(asm.shape(), shape)


def test_subassembly_items(asm_data):
    n, asm, shape = asm_data
    seen = {}

    for item in asm:
        assert item.name not in seen
        seen[item.name] = item

    for i in range(1, n + 1):
        name = "row_{}".format(i)
        assert name in seen
        del seen[name]

    assert "shape_0" in seen
    del seen["shape_0"]

    assert len(seen) == 0
