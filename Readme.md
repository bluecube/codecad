A toy programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python <strike>and using the awesomeness of Theano for performance (we'll see about that)</strike>.

This branch is an attempt to get rid of Theano's slow compilation times and
work directly with OpenCL instead.

Main sources so far:
- https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
- http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
- http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

## To do
-  Extrusions
  - [ ] With scaling
  - [ ] Twisted extrusion
  - [ ] Helix
  - [ ] Along a general curve
- [ ] Flip primitive
- [ ] Ellipsoid
- [ ] NACA airfoils
- [ ] Involute gears
- [ ] Assemblies (operations on combined shapes that propagate back to components)
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
- Fix problems created by the opencl conversion
  - [X] STL export
  - [X] Matplotlib slice
  - [ ] Volume and centroid
  - [ ] Animations
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
  - [ ] Involute gears
  - [ ] Mecha generator
