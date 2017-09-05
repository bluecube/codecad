from setuptools import setup, find_packages

setup(name = "codecad",
      version = "0.1.0",
      description = "OpenCL powered programming CAD",
      keywords = "cad csg",
      url = "http://github.com/bluecube/codecad",
      author = "Kuba Marek",
      author_email = "blue.cube@seznam.cz",
      license = "GPL",
      packages = find_packages(),
      package_data = {
        "": ["*.cl", "*.h"],
        },
      install_requires = [
        "numpy",
        "pyopencl",
        "pillow",
        "pymcubes",
        "numpy-stl",
        "py-flags",
        ],
      setup_requires = [
        "pytest-runner",
        ],
      tests_require = [
        "pytest",
        "trimesh",
        ],
      python_requires = "~=3.4",

      zip_safe = False,
      )
