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

- Primitives wishlist
  - [X] Offset
  - [ ] Ellipsoid
  - [X] Involute curve
  -  Extrusions
    - [ ] With scaling
    - [ ] Twisted extrusion
    - [ ] Along a <strike>general curve</strike> spline
  - [ ] Projection to plane
  - [ ] Slice by a plane (Correct one, not just set Z to 0)
  - [ ] 2D closed spline
  - [ ] Polygonal model
    - [ ] STL import
  - [ ] Swipe 3D object (allows extrusion from or up to a general surface)
- [ ] Assemblies (operations on combined shapes that propagate back to components) :four:
  - [ ] Exploded view
- [ ] Modules (pieces of geometry with both positive and negative space)
- Fancy raycaster effects
    - [ ] Axes
    - [ ] Shadows, ambient occlusion
    - [ ] Experiment with three point lighting
    - [ ] Transparent PNGs
- [ ] Successive approximation of the geometry :one: (needed for volume and centroid)
- [ ] Better geometry representation in Jupyter notebook (using three.js?)
  - [ ] Figure out simple way to use virtualenv in Jupyter
  - [ ] Use this to make a documentation notebook
- [ ] Add setup.py
- [ ] Fix rounded unions! :three:
- [ ] Parametrization of the model
- [ ] tests! :one:
- [ ] Models with high dynamic range (1m vs 1nm in the same model)
  - [ ] Multiprecision? `float` for initial approximation, `double` for final details
- [ ] Boundary conditions -- unions of touching objects
- [ ] OpenSCAD csg export
- [ ] Colors / materials
- [ ] Asynchronous API :four:
- [ ] Get rid of PyMCubes use GPU for triangulation (dual contouring?) :four:
- [ ] Clean up progress reporting and script/application/notebook interfaces
- [ ] Mass properties separated from subdivision
 - [ ] Volume
 - [ ] Centroid
 - [ ] Inertia matrix?
- Fix problems created by the opencl conversion
  - [X] STL export
  - [X] Matplotlib slice
  - [X] Volume and centroid
  - [ ] Animations
  - [ ] Svg export :two:
  - [ ] Tests :one:
    - [ ] Doctests
    - [ ] Test examples
- Optimization ideas
  - [ ] Merge `transformation_to` and `transformation_from` blocks
  - [ ] Don't output no-op transformations
  - [ ] Pass scaling operations through into the primitives
    - [ ] Turn sphere and circle rotations into no-ops
  - [ ] Separate translation only and rotation only transformations
  - [ ] Separate 2d transformations
  - [ ] Convert 90° rotations and -1 scaling to vector component shuffling and negations
  - [ ] Include bounding volumes in instruction stream and use it to clip the computation
  - [ ] Try speeding up subdivision code (and others) with prefix sums
- Things to model with it
  - [ ] _mystery project_ (NACA airfoils, meshed internal structure)
  - [ ] Box generator (Assemblies)
  - [X] Involute gears
  - [ ] Mecha generator
- SDF problems
  - [ ] Empty intersection of two objects still has gradients pointing somewhere random
  - [ ] `object.offset(-1).offset(1)` gives the original object, without rounded corners
