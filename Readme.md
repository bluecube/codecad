A toy programming CAD based on signed distance functions
========================================================

Inspired by OpenSCAD and ImplicitCad, implemented in Python and using the awesomeness of Theano for performance (we'll see about that).

Main sources so far:
https://christopherolah.wordpress.com/2011/11/06/manipulation-of-implicit-functions-with-an-eye-on-cad/
http://9bitscience.blogspot.com/2013/07/raymarching-distance-fields_14.html
http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

To do
-----
(just the main things, roughly in order)

- [ ] Get ray caster working properly (not just black on white)
- [ ] Matrix transformations
- [ ] Extrusions
- [ ] Marching cubes & STL output
- [ ] Figure out converting objects given by parametric equations to implicit functions (should be easy, right? :-D)
- [ ] Use the whole thing to model the _mystery project_.
