A toy programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python, using OpenCL for
computing power.

Main sources so far:
- https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
- http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
- http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

## To do
(in no particular order)

- More primitives
  - [X] Offset
  - [ ] Flip
  - [ ] Ellipsoid
  - [ ] NACA airfoils
  - [X] Involute curve
  -  Extrusions
    - [ ] With scaling
    - [ ] Twisted extrusion
    - [ ] Along a general curve
- [ ] Assemblies (operations on combined shapes that propagate back to components)
- [ ] Modules (pieces of geometry with both positive and negative space)
- Fancy raycaster effects
    - [ ] Axes
    - [ ] Shadows, ambient occlusion
    - [ ] Experiment with three point lighting
    - [ ] Transparent PNGs
- [ ] Successive approximation of the geometry
- [ ] Better geometry representation in IPython using three.js
  - [ ] Use this to make a documentation notebook
- [ ] Add setup.py
- [ ] *Fix rounded unions!*
- [ ] Parametrization of the model
- [ ] *tests!*
- [ ] Models with high dynamic range (1m vs 1nm in the same model)
- [ ] Boundary conditions -- unions of touching objects
- [ ] OpenSCAD csg export
- [ ] Colors / materials
- [ ] Asynchronous API
- Fix problems created by the opencl conversion
  - [X] STL export
  - [X] Matplotlib slice
  - [ ] Volume and centroid
  - [ ] Animations
  - [ ] Svg export
  - [ ] Tests
- Optimization ideas
  - [ ] Merge `transformation_to` and `transformation_from` blocks
  - [ ] Don't output no-op transformations
  - [ ] Pass scaling operations through into the primitives -- combo with previous
  - [ ] Separate translation only and rotation only transformations
  - [ ] Separate 2d transformations
  - [ ] Turn sphere and circle rotations into no-ops
  - [ ] Convert 90° rotations to vector component shuffling and negations (related to flip primitive)
- Things to model with it
  - [ ] _mystery project_ (NACA airfoils, meshed internal structure)
  - [ ] Box generator (Assemblies)
  - [X] Involute gears
  - [ ] Mecha generator
- SDF problems
  - [ ] Empty intersection of two objects still has gradients pointing somewhere random
  - [ ] `object.offset(-1).offset(1)` gives the original object, without rounded corners
