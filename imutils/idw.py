import numpy as np
from scipy.spatial import cKDTree

__date__ = "2015-04-25"
__version__ = "0.1"
__vdate__ = '2015-04-25'
__author__ = "Mihai Cara"


__all__ = [ 'ShepardIDWInterpolator' ]


class ShepardIDWInterpolator(object):
    """
Class to perform Inverse Distance Weighted (IDW) interpolation on unstructured
data using a modified version of the
`Shepard's method <http://en.wikipedia.org/wiki/Inverse_distance_weighting>`_
(see Notes section for details).

Examples
--------
This class can can be instantiated using the following syntax::

    >>> import imutils.ShepardIDWInterpolator as idw
    >>> import numpy as np

Example of interpolating 1D data::

    >>> x = np.random.random(100)
    >>> y = np.sin(2.0*x)
    >>> f = idw(x, y)
    >>> f(0.4)
    0.38939783923570831
    >>> np.sin(2.0*0.4)
    0.38941834230865052
    >>> xi = np.random.random(10)
    >>> print(xi)
    [ 0.36959095  0.13393148  0.06462452  0.12486564  0.85216626  0.26699299
      0.18332824  0.07311128  0.41488567  0.75356603]
    >>> f(xi)
    array([ 0.6908391 ,  0.25915542,  0.12856382,  0.2471138 ,  0.98924021,
            0.51959816,  0.35847361,  0.16208274,  0.73641671,  0.9979987 ])
    >>> np.sin(2.0*xi)
    array([ 0.67368354,  0.2646712 ,  0.12888948,  0.24714359,  0.99109728,
            0.50896845,  0.35849616,  0.14570204,  0.73777703,  0.99797412])

NOTE: In the last example, ``xi`` may be a ``Nx1`` array instead of a 1D
vector.

Example of interpolating 2D data::
    >>> x = np.random.rand(1000,2)
    >>> v = np.sin(x[:,0]+x[:,1])
    >>> f = idw(x, v)
    >>> f([0.5,0.6])
    0.88677703934471241
    >>> np.sin(0.5+0.6)
    0.89120736006143542

Notice that when a single coordinate is passed as an argument to the
interpolator, then a single (interpolated) value is returned (instead of a
1D vector of values).

Parameters
----------
coord : int, float, 1D vector, or NxM-array-like of int or float
    Coordinates of the known data points. In general, it is expected that
    these coordinates are in a form of a NxM-like array where N is the number
    of points and M is dimention of the coordinate space. When N=1 (1D space),
    then the `coord` parameter may be entered as a 1D-like array (vector) or,
    if only one data point is available, `coord` can be a simple number
    representing the 1D coordinate of the data point.

    .. note::
        If dimensionality of the `coord` argument is larger than 2, e.g.,
        if it is of the form N1xN2xN3x...xNnxM then it will be
        flattened down to the last dimention to form an array of size NxM
        where N=N1*N2*...Nn.

vals : int, float, complex, or 1D vector of int, float, or complex
    Values of the data points corresponding to each point coordinate provided
    in `coord`. In general a 1D-array like structure is expected.
    When a single data point is available, then `vals` can be a
    scalar (int, float, or complex).

    .. note::
        If dimensionality of `vals` is larger than one then it will be
        flattened.

weights : None, int, float, complex, or 1D vector of int, float, or complex (Default = None)
    Weights to be associated with each data point value. These weights, if
    provided, will be combined with inverse distance weights (see Notes
    section for details). When `weights` is `None` (default), then only IDW
    will be used. When provided, this input parameter must be of the same form
    as `vals`.

leafsize : float
    The number of points at which the k-d tree algorithm switches over
    to brute-force. `leafsize` must be positive. See `scipy.spacial.cKDTree`
    for further information.


NOTES
-----
The interpolator provided by `ShepardIDWInterpolator` uses a slightly modified
`Shepard's method <http://en.wikipedia.org/wiki/Inverse_distance_weighting>`_.
The essential difference is the introduction of a "regularization parameter
``r`` that is used when computing the inverse distance weights:

.. math::
    w_i = 1 / (d(x,x_i)^p+r)

By supplying a positive regularization parameter, one can avoid singularities
at locations of the data points as well as control the "smoothness" of
the interpolation (e.g., make weights of the neighors less varied). The
"smoothness" of interpolation can also be controlled by the
power parameter ``p``.

    """
    def __init__(self, coord, vals, weights=None, leafsize=10):

        coord = np.asarray(coord)
        clen = len(coord.shape)

        if clen == 1:
            # assume we have a list of 1D coordinates
            coord = np.reshape(coord, (coord.shape[0], 1))

        elif clen == 0:
            # assume we have a single 1D coordinate
            coord = np.array(coord, copy=False, ndmin=2)

        elif clen != 2:
            #raise ValueError("Input coordinate(s) must be an array-like "
                             #"object of dimensionality no larger than 2.")
            coord = np.reshape(coord, (-1, coord.shape[-1]))

        vals = np.array(vals, copy=False, ndmin=1).flatten()

        ncoord = coord.shape[0]
        ncoord_dim = coord.shape[1]
        nval = vals.shape[0]

        if ncoord != nval:
            raise ValueError("The number of values must match the "
                             "number of coordinates.")

        if ncoord < 1:
            raise ValueError("Too few data points.")

        ## Do a reasonable effort to detect incorrectly shaped data arrays
        ## including coordinates of uneven length, or coordinates/values
        ## of higher than expected dimentionality.
        #if coord.dtype == np.object:
            #if len(set(map(len, coords))):
                #raise TypeError("The input coordinates either have unequal "
                                #"dimensionalities or unsupported type.")

        if weights is not None:
            weights = np.array(weights, copy=False, ndmin=1).flatten()

            if ncoord != weights.shape[0]:
                raise ValueError("The number of weights must match the "
                                 "number of coordinates.")

            if np.any(weights < 0.0):
                raise ValueError("All weights must be non-negative numbers.")

        self.kdtree = cKDTree(coord, leafsize=leafsize)
        self.ncoord_dim = ncoord_dim
        self.ncoord = ncoord
        self.vals = vals
        self.weights = weights


    def __call__(self, pts, nbr=8, eps=0.0, p=1, reg=0.0, confdist=1e-12, dtype=np.float):
        """
__call__(self, pts, nbr=8, eps=0.0, p=1, reg=0.0, confdist=1e-12, dtype=np.float)
Evaluate interpolator at given points.

Parameters
----------
pts : int, float, 1D vector, or NxM-array-like of int or float
    Coordinates of the point(s) at which the interpolator should be evaluated.
    In general, it is expected that these coordinates are in a form of a
    NxM-like array where N is the number of points and M is dimention of the
    coordinate space. When N=1 (1D space),
    then the `pts` parameter may be entered as a 1D-like array (vector) or,
    if only one data point is available, `pts` can be a simple number
    representing the 1D coordinate of the data point.

    .. note::
        If dimensionality of the `pts` argument is larger than 2, e.g.,
        if it is of the form N1xN2xN3x...xNnxM then it will be
        flattened down to the last dimention to form an array of size NxM
        where N=N1*N2*...Nn.

    .. warning::
        The dimensionality of coordinate space of the `pts` must match
        the dimensionality of the coordinates used during the initialization
        of the interpolator.

nbr : int (Default = 8)
    Maximum number of closest neighbors to be used during the interpolation.

eps : float (Default = 0.0)
    Use approximate nearest neighbors; the kth neighbor is guaranteed to be
    no further than (1+eps) times the distance to the real kth nearest neighbor.
    See `scipy.spacial.cKDTree.query` for further information.

p : int, float (Default = 1)
    Power parameter of the inverse distance.

reg : float (Default = 0.0)
    Regularization parameter. It may be used to control smoothness of the
    interpolator. See Notes section in `~ShepardIDWInterpolator` for more
    details.

confdist : float (Default = 1.0e-12)
    Confusion distance below which the interpolator should use the value of
    the closest data point instead of attempting to interpolate. This
    is used to avoid singularities at the known data points especially if
    `reg` is 0.0.

dtype : data-type (Default = numpy.float)
    The type of the output interpolated values. If `None` then the type will
    be inferred from the type of the `vals` parameter used during the
    initialization of the interpolator.

        """
        nbr = int(nbr)
        if nbr < 1:
            raise ValueError("Number of neighbours must be a positive integer.")

        if confdist is not None and confdist <= 0.0:
            confdist = None

        pts = np.asarray(pts)
        npts_dim = len(pts.shape)

        if npts_dim == 1:
            # assume we have a single point in the npts_dim-dimentional space
            if self.ncoord_dim != 1 and pts.shape[-1] != self.ncoord_dim:
                raise ValueError("Input point was provided as a 1D vector "
                                 "but its lengths does not match the "
                                 "dimentionality of the coordinate space "
                                 "used when the interpolant was constructed.")

        elif npts_dim == 0:
            # assume we have a single 1D coordinate
            if self.ncoord_dim != 1:
                raise ValueError("Input point dimensionality does not match "
                                 "the dimentionality of the coordinate space "
                                 "used when the interpolant was constructed.")

        elif npts_dim != 2:
            raise ValueError("Input point coordinate(s) must be an array-like "
                             "object of dimensionality no larger than 2.")

        pts = np.reshape(pts, (-1,self.ncoord_dim))
        npts = pts.shape[0]

        d, idx = self.kdtree.query(pts, k=nbr, eps=eps)

        if nbr == 1:
            return self.vals[idx]

        if dtype is None:
            dtype = self.vals.dtype

        ival = np.zeros(npts, dtype=dtype)

        for k in range(npts):
            valid = np.isfinite(d[k])
            idk = idx[k][valid]
            dk = d[k][valid]

            if dk.shape[0] == 0:
                ival[k] = np.nan
                continue

            if confdist:
                # check if we are close to a known data point
                confused = dk <= confdist
                if np.any(confused):
                    ival[k] = self.vals[idk[confused][0]]
                    continue

            w = 1.0 / (dk**p + reg)
            if self.weights is not None:
                w *= self.weights[idk]

            wtot = np.sum(w)
            if wtot > 0.0:
                ival[k] = np.dot(w, self.vals[idk]) / wtot
            else:
                ival[k] = np.nan

        return (ival if npts_dim > 0 else ival[0])
