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
        "": "*.cl",
        },
      install_requires = [
        "numpy",
        "pyopencl",
        "pillow",
        "pymcubes",
        "numpy-stl",
        ],
      setup_requires = [
        "pytest-runner",
        ],
      tests_require = [
        "pytest",
        "trimesh",
        ],

      zip_safe = False,
      )
