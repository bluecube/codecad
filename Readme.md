A toy programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python and using the awesomeness of Theano for performance (we'll see about that).

Main sources so far:
- https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
- http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
- http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

## To do
- [X] Get ray caster working properly (not just black on white) (dot product shading works)
  - <strike>[ ] Try to avoid calculating the normals explicitly, instead take them from neighboring pixel distances</strike> (That probably wouldn't have worked. Now we're calculating a dot product between a normal at a point and a vector directly using just a single extra evaluation and this looks good enough.)
- [X] Matrix transformations (no matrices, but translation, rotation and uniform scaling work now)
-  Extrusions
  - [X] Revolve
  - [ ] Twisted extrusion
  - [ ] Along a general curve
- [X] Marching cubes & STL output (initial version is done)
- [ ] NACA airfoils
- [ ] Involute gears
- [ ] Shape wrappers
- [ ] Assemblies (operations on combined shapes that propagate back to components)
- Animations
  - [ ] Interface
  - [ ] Rendering
- [ ] Show axes in raytraced image
- <strike>[ ] Function for successive approximation of the geometry</strike> (it was super slow, maybe later)
  - [X] Use it to calculate model volume and centroid (done using a much simpler algorithm)
  - <strike>[ ] Use it to improve STL export</strike>
- [ ] Better geometry representation in IPython using three.js
  - [ ] Use this to make a documentation notebook
- [ ] Add setup.py
- [ ] *Fix rounded unions!*
- Things to model with it:
  - [ ] _mystery project_ (NACA airfoils, bezier curve based extrusions, meshed internal structure)
  - [ ] Box generator (Assemblies)
  - [ ] Involute gears
