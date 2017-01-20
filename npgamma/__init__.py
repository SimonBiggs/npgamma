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
It computes 1, 2, or 3 dimensional gamma with arbitrary gird sizes while
interpolating on the fly.
This module makes use of some of the ideas presented within
<http://dx.doi.org/10.1118/1.2721657>.

It needs to be noted that this code base has not yet undergone sufficient
independent validation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Process, Queue


def _run_input_checks(
        coords_reference, dose_reference,
        coords_evaluation, dose_evaluation):
    """Check user inputs."""
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
                    "Can only use numpy arrays as input for one dimensional "
                    "gamma."
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

    return coords_reference, coords_evaluation


def _calculate_coordinates_kernel(
        distance, num_dimensions, distance_step_size):
    """Determine the coodinate shifts required.

    Coordinate shifts are determined to check the reference dose for a
    given distance, dimension, and step size
    """
    if num_dimensions == 1:
        if distance == 0:
            x_coords = np.array([0])
        else:
            x_coords = np.array([distance, -distance])

        return (x_coords,)

    elif num_dimensions == 2:
        amount_to_check = np.floor(
            2 * np.pi * distance / distance_step_size) + 2
        theta = np.linspace(0, 2*np.pi, amount_to_check + 1)[:-1:]
        x_coords = distance * np.cos(theta)
        y_coords = distance * np.sin(theta)

        return (x_coords, y_coords)

    elif num_dimensions == 3:
        number_of_rows = np.floor(
            np.pi * distance / distance_step_size) + 2
        elevation = np.linspace(0, np.pi, number_of_rows)
        row_radii = distance * np.sin(elevation)
        row_circumference = 2 * np.pi * row_radii
        amount_in_row = np.floor(
            row_circumference / distance_step_size) + 2

        x_coords = []
        y_coords = []
        z_coords = []
        for i, phi in enumerate(elevation):
            azimuth = np.linspace(0, 2*np.pi, amount_in_row[i] + 1)[:-1:]
            x_coords.append(distance * np.sin(phi) * np.cos(azimuth))
            y_coords.append(distance * np.sin(phi) * np.sin(azimuth))
            z_coords.append(distance * np.cos(phi) * np.ones_like(azimuth))

        return (
            np.hstack(x_coords), np.hstack(y_coords), np.hstack(z_coords))

    else:
        raise Exception("No valid dimension")


def _calculate_min_dose_difference(
        num_dimensions, mesh_coords_evaluation, to_be_checked,
        reference_interpolation, dose_evaluation,
        coordinates_at_distance_kernel):
    """Determine the minimum dose difference.

    Calculated for a given distance from each evaluation point.
    """
    coordinates_at_distance = []
    for i in range(num_dimensions):
        coordinates_at_distance.append(np.array(
            mesh_coords_evaluation[i][to_be_checked][None, :] +
            coordinates_at_distance_kernel[i][:, None])[:, :, None])

    all_points = np.concatenate(coordinates_at_distance, axis=2)

    dose_difference = np.array([
        reference_interpolation(points) -
        dose_evaluation[to_be_checked] for
        points in all_points
    ])
    min_dose_difference = np.min(np.abs(dose_difference), axis=0)

    return min_dose_difference


def _step_through_slices(
        current_thread_set, all_checks, num_dimensions, mesh_coords_evaluation,
        to_be_checked, reference_interpolation, dose_evaluation,
        coordinates_at_distance_kernel, output):
    min_dose_difference = (np.nan * np.ones_like(all_checks[0]))

    for current_slice in current_thread_set:
        current_to_be_checked = np.zeros_like(to_be_checked).astype(bool)
        current_to_be_checked[[
            item[current_slice] for
            item in all_checks]] = True

        assert np.all(to_be_checked[current_to_be_checked])

        min_dose_difference[np.sort(current_slice)] = (
            _calculate_min_dose_difference(
                num_dimensions, mesh_coords_evaluation, current_to_be_checked,
                reference_interpolation, dose_evaluation,
                coordinates_at_distance_kernel))

    output.put(min_dose_difference)


def _calculate_min_dose_difference_by_slice(
        max_concurrent_calc_points, num_threads,
        num_dimensions, mesh_coords_evaluation, to_be_checked,
        reference_interpolation, dose_evaluation,
        coordinates_at_distance_kernel, **kwargs):
    """Determine minimum dose differences.

    Calculation is made with the evaluation set divided into slices. This
    enables less RAM usage.
    """
    all_checks = np.where(to_be_checked)

    num_slices = num_threads * (np.floor(
        len(coordinates_at_distance_kernel[0]) *
        len(all_checks[0]) / max_concurrent_calc_points) + 1)

    index = np.arange(len(all_checks[0]))
    np.random.shuffle(index)
    sliced = np.array_split(index, num_slices)

    thread_sets_of_slices = np.array_split(sliced, num_threads)

    output_queue = Queue()

    for current_thread_set in thread_sets_of_slices:
        Process(
            target=_step_through_slices,
            args=(
                current_thread_set,
                all_checks, num_dimensions, mesh_coords_evaluation,
                to_be_checked, reference_interpolation, dose_evaluation,
                coordinates_at_distance_kernel, output_queue)).start()

    min_dose_difference = (np.nan * np.ones_like(all_checks[0]))

    for i in range(num_threads):
        result = output_queue.get()
        ref = np.invert(np.isnan(result))
        min_dose_difference[ref] = result[ref]

    assert np.all(np.invert(np.isnan(min_dose_difference)))

    return min_dose_difference


def _calculation_loop(**kwargs):
    """Iteratively calculates gamma at increasing distances."""
    gamma_valid = np.ones_like(kwargs['dose_evaluation']).astype(bool)
    running_gamma = np.inf * np.ones_like(kwargs['dose_evaluation'])
    distance = 0

    while True:
        to_be_checked = (
            kwargs['dose_valid'] & gamma_valid)

        for i in range(kwargs['num_dimensions']):
            bounds_test = (
                (
                    kwargs['mesh_coords_evaluation'][i] >=
                    np.min(kwargs['coords_reference'][i]) + distance) &
                (
                    kwargs['mesh_coords_evaluation'][i] <=
                    np.max(kwargs['coords_reference'][i]) - distance))

            to_be_checked = to_be_checked & bounds_test

        coordinates_at_distance_kernel = _calculate_coordinates_kernel(
            distance, kwargs['num_dimensions'], kwargs['distance_step_size'])
        min_dose_difference = _calculate_min_dose_difference_by_slice(
            to_be_checked=to_be_checked,
            coordinates_at_distance_kernel=coordinates_at_distance_kernel,
            **kwargs)

        gamma_at_distance = np.sqrt(
            min_dose_difference ** 2 / kwargs['dose_threshold'] ** 2 +
            distance ** 2 / kwargs['distance_threshold'] ** 2)

        running_gamma[to_be_checked] = np.min(
            np.vstack((
                gamma_at_distance, running_gamma[to_be_checked]
            )), axis=0)

        gamma_valid = running_gamma > distance / kwargs['distance_threshold']

        distance += kwargs['distance_step_size']

        if (
                (np.sum(to_be_checked) == 0) |
                (distance > kwargs['maximum_test_distance'])):
            break

    running_gamma[np.isinf(running_gamma)] = np.nan
    return running_gamma


def calc_gamma(coords_reference, dose_reference,
               coords_evaluation, dose_evaluation,
               distance_threshold, dose_threshold,
               lower_dose_cutoff=0, distance_step_size=None,
               maximum_test_distance=np.inf,
               max_concurrent_calc_points=np.inf,
               num_threads=1):
    """Compare two dose grids with the gamma index.

    Args:
        coords_reference (tuple): The reference coordinates.
        dose_reference (np.array): The reference dose grid.
        coords_evaluation (tuple): The evaluation coordinates.
        dose_evaluation (np.array): The evaluation dose grid.
        distance_threshold (float): The gamma distance threshold. Units must
            match of the coordinates given.
        dose_threshold (float): An absolute dose threshold.
            If you wish to use 3% of maximum reference dose input
            np.max(dose_reference) * 0.03 here.
        lower_dose_cutoff (:obj:`float`, optional): The lower dose cutoff below
            which gamma will not be calculated.
        distance_step_size (:obj:`float`, optional): The step size to use in
            within the reference grid interpolation. Defaults to a tenth of the
            distance threshold as recommended within
            <http://dx.doi.org/10.1118/1.2721657>.
        maximum_test_distance (:obj:`float`, optional): The distance beyond
            which searching will stop. Defaults to np.inf. To speed up
            calculation it is recommended that this parameter is set to
            something reasonable such as 2*distance_threshold

    Returns:
        gamma (np.array): The array of gamma values the same shape as that
            given by the evaluation coordinates and dose.
    """
    coords_reference, coords_evaluation = _run_input_checks(
        coords_reference, dose_reference,
        coords_evaluation, dose_evaluation)

    if distance_step_size is None:
        distance_step_size = distance_threshold / 10

    reference_interpolation = RegularGridInterpolator(
        coords_reference, np.array(dose_reference)
    )

    dose_evaluation = np.array(dose_evaluation)
    dose_valid = dose_evaluation >= lower_dose_cutoff
    mesh_coords_evaluation = np.meshgrid(*coords_evaluation, indexing='ij')

    kwargs = {
        "coords_reference": coords_reference,
        "num_dimensions": len(coords_evaluation),
        "reference_interpolation": reference_interpolation,
        "dose_evaluation": dose_evaluation,
        "dose_valid": dose_valid,
        "mesh_coords_evaluation": mesh_coords_evaluation,
        "distance_threshold": distance_threshold,
        "dose_threshold": dose_threshold,
        "distance_step_size": distance_step_size,
        "max_concurrent_calc_points": max_concurrent_calc_points,
        "maximum_test_distance": maximum_test_distance,
        "num_threads": num_threads}

    gamma = _calculation_loop(**kwargs)

    return gamma
