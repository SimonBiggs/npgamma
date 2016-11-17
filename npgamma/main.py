# Copyright (C) 2015 Simon Biggs
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# http://www.gnu.org/licenses/.

"""Compare two dose grids with the gamma index.

This module is a python implementation of the gamma index.
It computes 1, 2, or 3 dimensional gamma with arbitrary gird sizes while interpolating on the fly.
This module makes use of some of the ideas presented within <http://dx.doi.org/10.1118/1.2721657>.

It needs to be noted that this code base has not yet undergone sufficient independent validation.
"""

import numpy as np

from scipy.interpolate import RegularGridInterpolator


# Pulled from future numpy due to gapps numpy meshgrid not working for 3D.
def meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from coordinate vectors.
    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.
    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory.
        Default is False.
        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.
        .. versionadded:: 1.7.0
    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.
    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::
        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]
        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]
    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.
    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])
    `meshgrid` is very useful to evaluate functions on a grid.
    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)
    """
    ndim = len(xi)

    copy_ = kwargs.pop('copy', True)
    sparse = kwargs.pop('sparse', False)
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x in enumerate(xi)]

    shape = [x.size for x in output]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


def calc_gamma(coords_reference, dose_reference,
               coords_evaluation, dose_evaluation,
               distance_threshold, dose_threshold,
               lower_dose_cutoff=0, distance_step_size=None,
               maximum_test_distance=np.inf):
    """Compare two dose grids with the gamma index.

    Args:
        coords_reference (tuple): The reference coordinates.
        dose_reference (np.array): The reference dose grid.
        coords_evaluation (tuple): The evaluation coordinates.
        dose_evaluation (np.array): The evaluation dose grid.
        distance_threshold (float): The gamma distance threshold. Units must match of the
            coordinates given.
        dose_threshold (float): An absolute dose threshold.
            If you wish to use 3% of maximum reference dose input
            np.max(dose_reference) * 0.03 here.
        lower_dose_cutoff (:obj:`float`, optional): The lower dose cutoff below which gamma
            will not be calculated.
        distance_step_size (:obj:`float`, optional): The step size to use in within the
            reference grid interpolation. Defaults to a tenth of the distance threshold as
            recommended within <http://dx.doi.org/10.1118/1.2721657>.
        maximum_test_distance (:obj:`float`, optional): The distance beyond which searching
            will stop. Defaults to np.inf. To speed up calculation it is recommended that this
            parameter is set to something reasonable such as 2*distance_threshold

    Returns:
        gamma (np.array): The array of gamma values the same shape as that given by the evaluation
            coordinates and dose.
        """
    if (
            not isinstance(coords_evaluation, tuple) or
            not isinstance(coords_reference, tuple)):
        if (
                isinstance(coords_evaluation, np.ndarray) and
                isinstance(coords_reference, np.ndarray)):
            if (
                    len(np.shape(coords_evaluation)) == 1 and
                    len(np.shape(coords_reference)) == 1):

                coords_evaluation = (coords_evaluation,)
                coords_reference = (coords_reference,)

            else:
                raise Exception(
                    "Can only use numpy arrays as input for one dimensional gamma."
                    )
        else:
            raise Exception(
                "Input coordinates must be inputted as a tuple, for "
                "one dimension input is (x,), for two dimensions, (x, y),  "
                "for three dimensions input is (x, y, z).")

    reference_coords_shape = tuple([len(item) for item in coords_reference])
    if reference_coords_shape != np.shape(dose_reference):
        raise Exception(
            "Length of items in coords_reference does not match the shape of "
            "dose_reference")

    evaluation_coords_shape = tuple([len(item) for item in coords_evaluation])
    if evaluation_coords_shape != np.shape(dose_evaluation):
        raise Exception(
            "Length of items in coords_evaluation does not match the shape of "
            "dose_evaluation")

    if not (len(np.shape(dose_evaluation)) ==
            len(np.shape(dose_reference)) ==
            len(coords_evaluation) ==
            len(coords_reference)):
        raise Exception(
            "The dimensions of the input data do not match")


    gamma_calculation = GammaCalculation(
        coords_reference, dose_reference,
        coords_evaluation, dose_evaluation,
        distance_threshold, dose_threshold,
        lower_dose_cutoff=lower_dose_cutoff,
        distance_step_size=distance_step_size,
        maximum_test_distance=maximum_test_distance)


    return gamma_calculation.gamma


class GammaCalculation():
    """Compare two dose grids with the gamma index.

    Args:
        coords_reference (tuple): The reference coordinates.
        dose_reference (np.array): The reference dose grid.
        coords_evaluation (tuple): The evaluation coordinates.
        dose_evaluation (np.array): The evaluation dose grid.
        distance_threshold (float): The gamma distance threshold. Units must match of the
            coordinates given.
        dose_threshold (float): An absolute dose threshold.
            If you wish to use 3% of maximum reference dose input
            np.max(dose_reference) * 0.03 here.
        lower_dose_cutoff (:obj:`float`, optional): The lower dose cutoff below which gamma
            will not be calculated.
        distance_step_size (:obj:`float`, optional): The step size to use in within the
            reference grid interpolation. Defaults to a tenth of the distance threshold as
            recommended within <http://dx.doi.org/10.1118/1.2721657>.
        maximum_test_distance (:obj:`float`, optional): The distance beyond which searching
            will stop. Defaults to np.inf. To speed up calculation it is recommended that this
            parameter is set to something reasonable such as 2*distance_threshold

    Attribute:
        gamma (np.array): The array of gamma values the same shape as that given by the evaluation
            coordinates and dose.
        """
    def __init__(
            self,
            coords_reference, dose_reference,
            coords_evaluation, dose_evaluation,
            distance_threshold, dose_threshold,
            lower_dose_cutoff=0, distance_step_size=None,
            maximum_test_distance=np.inf):

        self.distance_threshold = distance_threshold
        self.dose_threshold = dose_threshold
        self.lower_dose_cutoff = lower_dose_cutoff

        if distance_step_size is None:
            self.distance_step_size = distance_threshold / 10
        else:
            self.distance_step_size = distance_step_size

        self.maximum_test_distance = maximum_test_distance

        self.dimension = len(coords_evaluation)

        if self.dimension == 3:
            self.coords_key = ['x', 'y', 'z']
        elif self.dimension == 2:
            self.coords_key = ['x', 'y']
        elif self.dimension == 1:
            self.coords_key = ['x']
        else:
            raise Exception("No valid dimension")

        self.dose_reference = np.array(dose_reference)
        self.dose_evaluation = np.array(dose_evaluation)

        self.coords_reference = coords_reference
        self.coords_evaluation = coords_evaluation

        self.within_bounds = {}
        for _, key in enumerate(self.coords_key):
            self.within_bounds[key] = np.array([])

        self.reference_interpolation = RegularGridInterpolator(
            self.coords_reference, self.dose_reference
        )

        self.dose_valid = self.dose_evaluation >= self.lower_dose_cutoff

        self.mesh_coords_evaluation = meshgrid(*self.coords_evaluation)

        self.gamma = self.calculate_gamma()


    def calculate_gamma(self):
        """Iteratively calculates gamma at increasing distances"""
        gamma_valid = np.ones_like(self.dose_evaluation).astype(bool)
        running_gamma = np.inf * np.ones_like(self.dose_evaluation)
        distance = 0

        while True:
            to_be_checked = (
                self.dose_valid & gamma_valid)

            for i, key in enumerate(self.coords_key):
                self.within_bounds[key] = (
                    (
                        self.mesh_coords_evaluation[i] >=
                        np.min(self.coords_reference[i]) + distance) &
                    (
                        self.mesh_coords_evaluation[i] <=
                        np.max(self.coords_reference[i]) - distance))

                to_be_checked = to_be_checked & self.within_bounds[key]

            min_dose_difference = self.min_dose_difference(to_be_checked, distance)

            gamma_at_distance = np.sqrt(
                min_dose_difference ** 2 / self.dose_threshold ** 2 +
                distance ** 2 / self.distance_threshold ** 2)

            running_gamma[to_be_checked] = np.min(
                np.vstack((
                    gamma_at_distance, running_gamma[to_be_checked]
                )), axis=0)

            gamma_valid = running_gamma > distance / self.distance_threshold

            distance += self.distance_step_size

            if (np.sum(to_be_checked) == 0) | (distance > self.maximum_test_distance):
                break

        running_gamma[np.isinf(running_gamma)] = np.nan
        return running_gamma


    def min_dose_difference(self, to_be_checked, distance):
        """Determines the minimum dose difference for a given distance from each evaluation point"""
        coordinates_at_distance_kernel = self.calculate_coordinates_kernel(distance)

        coordinates_at_distance = []
        for i, _ in enumerate(self.coords_key):
            coordinates_at_distance.append(np.array(
                self.mesh_coords_evaluation[i][to_be_checked][None, :] +
                coordinates_at_distance_kernel[i][:, None])[:, :, None])

        all_points = np.concatenate(coordinates_at_distance, axis=2)

        dose_difference = np.array([
            self.reference_interpolation(points) - self.dose_evaluation[to_be_checked] for
            points in all_points
        ])
        min_dose_difference = np.min(np.abs(dose_difference), axis=0)

        return min_dose_difference


    def calculate_coordinates_kernel(self, distance):
        """Determines the coodinate shifts required.

        Coordinate shifts are determined to check the reference dose for a given
        distance, dimension, and step size"""

        if self.dimension == 1:
            if distance == 0:
                x_coords = np.array([0])
            else:
                x_coords = np.array([distance, -distance])

            return (x_coords,)

        elif self.dimension == 2:
            amount_to_check = np.floor(
                2 * np.pi * distance / self.distance_step_size) + 2  # changed from 1 --> 2
            theta = np.linspace(0, 2*np.pi, amount_to_check + 1)[:-1:]
            x_coords = distance * np.cos(theta)
            y_coords = distance * np.sin(theta)

            return (y_coords, x_coords)

        elif self.dimension == 3:
            number_of_rows = np.floor(
                np.pi * distance / self.distance_step_size) + 2  # changed
            elevation = np.linspace(0, np.pi, number_of_rows)
            row_radii = distance * np.sin(elevation)
            row_circumference = 2 * np.pi * row_radii
            amount_in_row = np.floor(row_circumference / self.distance_step_size) + 2  # changed

            x_coords = []
            y_coords = []
            z_coords = []
            for i, phi in enumerate(elevation):
                azimuth = np.linspace(0, 2*np.pi, amount_in_row[i] + 1)[:-1:]
                x_coords.append(distance * np.sin(phi) * np.cos(azimuth))
                y_coords.append(distance * np.sin(phi) * np.sin(azimuth))
                z_coords.append(distance * np.cos(phi) * np.ones_like(azimuth))

            return (np.hstack(y_coords), np.hstack(x_coords), np.hstack(z_coords))

        else:
            raise Exception("No valid dimension")
