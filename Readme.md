A programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python, using OpenCL for
computing power.

## Roadmap

### Version 1.0.0
- ETA October 2017
- Feature set comparable to OpenSCAD
- Modeling in IPython or with standalone scripts
- Minimal performance optimizations
- Features required:
  - [ ] Extrusion with scaling
  - [ ] Twisted extrusion
  - [ ] Projection to plane
  - [ ] Polygonal model
    - [ ] STL import
  - [ ] Rounded unions
  - [ ] Boundary conditions -- unions of touching objects
  - [ ] Svg export

## Ideas
- Primitives
  - [ ] Ellipsoid
  - [X] Involute curve
  - [ ] Extrusions controlled by a spline
  - [ ] Slice by a plane (Correct one, not just set Z to 0)
  - [ ] 2D closed spline
  - [ ] Swipe 3D object (allows extrusion from or up to a general surface)
- [ ] Assemblies (operations on combined shapes that propagate back to components) :four:
  - [ ] Exploded view
- [ ] Modules (pieces of geometry with both positive and negative space)
- Fancy raycaster effects
    - [ ] Axes
    - [ ] Shadows, ambient occlusion
    - [ ] Experiment with three point lighting
    - [ ] Transparent PNGs
- [X] Successive approximation of the geometry (needed for volume and centroid)
- [ ] Better geometry representation in Jupyter notebook (using three.js?)
  - [ ] Use this to make a documentation notebook
- [X] Add setup.py
- [ ] Parametrization of the model
- [ ] Models with high dynamic range (1m vs 1nm in the same model)
  - [ ] Multiprecision? `float` for initial approximation, `double` for final details
- [ ] Colors / materials
- [ ] Asynchronous API
- [ ] Get rid of PyMCubes use GPU for triangulation (dual contouring?) :four:
- [ ] Clean up progress reporting and script/application/notebook interfaces
- [X] Mass properties separated from subdivision
 - [X] Volume
 - [X] Centroid
 - [X] Inertia tensor
- Testing
  - MORE TESTS FOR EVERYTHING
  - [ ] Setup Travis
  - [ ] Doctests
  - [ ] Test examples

## Optimization ideas
- [ ] Rematerialization of values in evaluate.
- [ ] Include bounding volumes in instruction stream and use it to clip the computation
- [X] Merge `transformation_to` and `transformation_from` blocks
- [ ] Don't output no-op transformations
- [ ] Pass scaling operations through into the primitives
  - [ ] Turn sphere and circle rotations into no-ops
- [ ] Separate translation only and rotation only transformations
- [ ] Separate 2d transformations
- [ ] Convert 90° rotations and -1 scaling to vector component shuffling and negations
- [ ] Try speeding up subdivision code (and others) with prefix sums
- [ ] Caching instruction buffers
- [ ] Try putting `restrict` everywhere in OpenCL

## SDF problems
  - [ ] Empty intersection of two objects still has gradients pointing somewhere random
  - [ ] `object.offset(-1).offset(1)` gives the original object, without rounded corners
