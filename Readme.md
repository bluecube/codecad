A toy programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python and using the awesomeness of Theano for performance (we'll see about that).

Main sources so far:
- https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
- http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
- http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

## To do
- [X] Get ray caster working properly (not just black on white) (dot product shading works)
  - [ ] Try to avoid calculating the normals explicitly, instead take them from neighboring pixel distances
- [X] Matrix transformations (no matrices, but translation, rotation and uniform scaling work now)
- [ ] Extrusions
- [X] Marching cubes & STL output (initial version is done)
- [ ] Figure out converting objects given by parametric equations to implicit functions (should be easy, right? :-D)
- [ ] Use the whole thing to model the _mystery project_.
- [ ] Function for successive approximation of the geometry
  - [ ] Use it to calculate model volume
  - [ ] Use it to improve STL export
- [ ] Better geometry representation in IPython
