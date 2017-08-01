## shamelessly copied from http://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
import numpy as np
from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=100):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals

if (__name__ == "__main__"):
    import matplotlib.pyplot as plt
    foil = np.loadtxt("/home/ashish/PycharmProjects/foil_solver/sample_foils/foil1.dat")
    points1 = np.array([[1, 0], [0.75, 0.25], [0.5, 0.5]])
    xs1, ys1 = bezier_curve(points1, nTimes=100)
    print(xs1, ys1)
    points2 = np.array([[0.5, 0.5], [0.25, 0.75], [0, 0]])
    xs2, ys2 = bezier_curve(points2, nTimes=100)
    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 2), ylim=(-1, 1))
    # plot bezier points
    ax.plot(xs1, ys1, "r-", xs2, ys2, "b-")
    # plot control points
    ax.plot(points1[:, 0], points1[:, 1], "r*")
    ax.plot(points2[:, 0], points2[:, 1], "b*")
    ax.plot(foil[:, 0], foil[:, 1], linewidth=2.0)
    plt.show()

