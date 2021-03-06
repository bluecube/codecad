import csv


def render_bom(obj, filename):
    # Resolution is unused

    with open(filename, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "count"])
        writer.writerows(
            [item.name, item.count] + item.part.attributes for item in obj.bom()
        )
