import codecad
import theano
import theano.tensor as T

def test_gradient():
    x = T.matrix("x")
    y = T.matrix("y")
    z = T.matrix("z")
    point = codecad.util.Vector(x, y, z)
    expr = x**2 + 2 * y**2

    args = codecad.util.Vector([[1, 2], [1, 2]],
                               [[1, 1], [2, 2]],
                               [[0, 0], [0, 0]])

    g = codecad.util.derivatives.gradient(expr, point)

    g_f = theano.function([x, y, z], g)
    g_num = g_f(*args)

    c = lambda a, b: codecad.util.check_close(a, b, 0.001)

    c(g_num[0][0][0], 2)
    c(g_num[0][0][1], 4)

    c(g_num[1][0][0], 4)
    c(g_num[1][0][1], 8)

    c(g_num[2][0][0], 6)
    c(g_num[2][0][1], 12)
