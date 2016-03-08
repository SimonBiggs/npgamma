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

import numpy as np

from scipy.interpolate import RegularGridInterpolator


def calc_gamma(coords_reference, dose_reference,
               coords_evaluation, dose_evaluation,
               distance_threshold, dose_threshold,
               lower_dose_cutoff=0, distance_step_size=None,
               maximum_test_distance=None):
    if (
            type(coords_evaluation) is not tuple or
            type(coords_reference) is not tuple):
        if (
                type(coords_evaluation) is np.ndarray and
                type(coords_reference) is np.ndarray):
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
                "one dimension input is (x,), for two dimensions, (y, x),  "
                "for three dimensions input is (y, x, z).")

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

    dimension = len(coords_evaluation)

    if dimension == 3:
        gamma = calc_gamma_3d(
            coords_reference, dose_reference,
            coords_evaluation, dose_evaluation,
            distance_threshold, dose_threshold,
            lower_dose_cutoff=lower_dose_cutoff,
            distance_step_size=distance_step_size,
            maximum_test_distance=maximum_test_distance)

    elif dimension == 2:
        gamma = calc_gamma_2d(
            coords_reference, dose_reference,
            coords_evaluation, dose_evaluation,
            distance_threshold, dose_threshold,
            lower_dose_cutoff=lower_dose_cutoff,
            distance_step_size=distance_step_size,
            maximum_test_distance=maximum_test_distance)
    elif dimension == 1:
        gamma = calc_gamma_1d(
            coords_reference, dose_reference,
            coords_evaluation, dose_evaluation,
            distance_threshold, dose_threshold,
            lower_dose_cutoff=lower_dose_cutoff,
            distance_step_size=distance_step_size,
            maximum_test_distance=maximum_test_distance)

    else:
        raise Exception("Unexpected dimension recieved")

    return gamma


def coords_to_check_1d(distance):
    if distance == 0:
        x = np.array([0])
    else:
        x = np.array([distance, -distance])

    return x


def coords_to_check_2d(distance, step_size):
    amount_to_check = np.floor(
        2 * np.pi * distance / step_size) + 1
    theta = np.linspace(0, 2*np.pi, amount_to_check + 1)[:-1:]
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)

    return x, y


def coords_to_check_3d(distance, step_size):
    number_of_rows = np.floor(
        np.pi * distance / step_size) + 1
    elevation = np.linspace(0, np.pi, number_of_rows)
    row_radii = distance * np.sin(elevation)
    row_circumference = 2 * np.pi * row_radii
    amount_in_row = np.floor(row_circumference / step_size) + 1

    x = []
    y = []
    z = []
    for i, phi in enumerate(elevation):
        azimuth = np.linspace(0, 2*np.pi, amount_in_row[i] + 1)[:-1:]
        x.append(distance * np.sin(phi) * np.cos(azimuth))
        y.append(distance * np.sin(phi) * np.sin(azimuth))
        z.append(distance * np.cos(phi) * np.ones_like(azimuth))

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)

    return x, y, z


def find_min_dose_difference_at_distance_1d(x_test, dose_test,
                                            reference_interpolation,
                                            distance):
    x_shift = coords_to_check_1d(
        distance)

    x_coords = x_test[None, :] + x_shift[:, None]

    all_points = np.concatenate(
        (x_coords[:, :, None],), axis=2)

    dose_difference = np.array([
        reference_interpolation(points) - dose_test for
        points in all_points
    ])
    min_dose_difference = np.min(np.abs(dose_difference), axis=0)

    return min_dose_difference


def find_min_dose_difference_at_distance_2d(x_test, y_test, dose_test,
                                            reference_interpolation,
                                            distance, step_size):
    x_shift, y_shift = coords_to_check_2d(
        distance, step_size)

    x_coords = x_test[None, :] + x_shift[:, None]
    y_coords = y_test[None, :] + y_shift[:, None]

    all_points = np.concatenate(
        (y_coords[:, :, None], x_coords[:, :, None]), axis=2)

    dose_difference = np.array([
        reference_interpolation(points) - dose_test for
        points in all_points
    ])
    min_dose_difference = np.min(np.abs(dose_difference), axis=0)

    return min_dose_difference


def find_min_dose_difference_at_distance_3d(x_test, y_test, z_test,
                                            dose_test, reference_interpolation,
                                            distance, step_size):
    x_shift, y_shift, z_shift = coords_to_check_3d(
        distance, step_size)

    x_coords = x_test[None, :] + x_shift[:, None]
    y_coords = y_test[None, :] + y_shift[:, None]
    z_coords = z_test[None, :] + z_shift[:, None]

    all_points = np.concatenate(
        (y_coords[:, :, None], x_coords[:, :, None], z_coords[:, :, None]),
        axis=2)

    dose_difference = np.array([
        reference_interpolation(points) - dose_test for
        points in all_points
    ])
    min_dose_difference = np.min(np.abs(dose_difference), axis=0)

    return min_dose_difference


def calc_gamma_1d(coords_reference, dose_reference,
                  coords_evaluation, dose_evaluation,
                  distance_threshold, dose_threshold,
                  lower_dose_cutoff=0, distance_step_size=None,
                  maximum_test_distance=None):
    if distance_step_size is None:
        distance_step_size = distance_threshold / 10

    if maximum_test_distance is None:
        maximum_test_distance = distance_threshold * 2

    reference_interpolation = RegularGridInterpolator(
        coords_reference, dose_reference
    )

    x_reference, = coords_reference

    x, = coords_evaluation
    xx = np.array(x)
    dose_evaluation = np.array(dose_evaluation)

    dose_valid = dose_evaluation > lower_dose_cutoff
    gamma_valid = np.ones_like(dose_evaluation).astype(bool)

    running_gamma = np.inf * np.ones_like(dose_evaluation)
    distance = 0

    while True:
        x_valid = (
            (xx >= np.min(x_reference) + distance) &
            (xx <= np.max(x_reference) - distance))

        to_be_checked = (
            dose_valid & x_valid & gamma_valid
        )

        min_dose_diff = find_min_dose_difference_at_distance_1d(
            xx[to_be_checked], dose_evaluation[to_be_checked],
            reference_interpolation,
            distance)

        gamma_at_distance = np.sqrt(
            min_dose_diff ** 2 / dose_threshold ** 2 +
            distance ** 2 / distance_threshold ** 2)

        running_gamma[to_be_checked] = np.min(
            np.vstack((
                    gamma_at_distance, running_gamma[to_be_checked]
                )), axis=0)

        gamma_valid = running_gamma > distance / distance_threshold

        distance += distance_step_size

        if (np.sum(to_be_checked) == 0) | (distance > maximum_test_distance):
            break

    return running_gamma


def calc_gamma_2d(coords_reference, dose_reference,
                  coords_evaluation, dose_evaluation,
                  distance_threshold, dose_threshold,
                  lower_dose_cutoff=0, distance_step_size=None,
                  maximum_test_distance=None):
    if distance_step_size is None:
        distance_step_size = distance_threshold / 10

    if maximum_test_distance is None:
        maximum_test_distance = distance_threshold * 2

    reference_interpolation = RegularGridInterpolator(
        coords_reference, dose_reference
    )

    y_reference, x_reference = coords_reference

    y, x = coords_evaluation
    xx, yy = np.meshgrid(x, y)
    dose_evaluation = np.array(dose_evaluation)

    dose_valid = dose_evaluation > lower_dose_cutoff
    gamma_valid = np.ones_like(dose_evaluation).astype(bool)

    running_gamma = np.inf * np.ones_like(dose_evaluation)

    distance = 0

    while True:
        x_valid = (
            (xx >= np.min(x_reference) + distance) &
            (xx <= np.max(x_reference) - distance))

        y_valid = (
            (yy >= np.min(y_reference) + distance) &
            (yy <= np.max(y_reference) - distance))

        to_be_checked = (
            dose_valid & x_valid & y_valid & gamma_valid
        )

        min_dose_diff = find_min_dose_difference_at_distance_2d(
            xx[to_be_checked], yy[to_be_checked],
            dose_evaluation[to_be_checked],
            reference_interpolation,
            distance, distance_step_size)

        gamma_at_distance = np.sqrt(
            min_dose_diff ** 2 / dose_threshold ** 2 +
            distance ** 2 / distance_threshold ** 2)

        running_gamma[to_be_checked] = np.min(
            np.vstack((
                    gamma_at_distance, running_gamma[to_be_checked]
                )), axis=0)

        gamma_valid = running_gamma > distance / distance_threshold

        distance += distance_step_size

        if (np.sum(to_be_checked) == 0) | (distance > maximum_test_distance):
            break

    return running_gamma


def calc_gamma_3d(coords_reference, dose_reference,
                  coords_evaluation, dose_evaluation,
                  distance_threshold, dose_threshold,
                  lower_dose_cutoff=0, distance_step_size=None,
                  maximum_test_distance=None):
    if distance_step_size is None:
        distance_step_size = distance_threshold / 10

    if maximum_test_distance is None:
        maximum_test_distance = distance_threshold * 2

    reference_interpolation = RegularGridInterpolator(
        coords_reference, dose_reference
    )

    y_reference, x_reference, z_reference = coords_reference

    y, x, z = coords_evaluation
    xx, yy, zz = np.meshgrid(x, y, z)
    dose_evaluation = np.array(dose_evaluation)

    dose_valid = dose_evaluation > lower_dose_cutoff
    gamma_valid = np.ones_like(dose_evaluation).astype(bool)

    running_gamma = np.inf * np.ones_like(dose_evaluation)

    distance = 0

    while True:
        x_valid = (
            (xx >= np.min(x_reference) + distance) &
            (xx <= np.max(x_reference) - distance))

        y_valid = (
            (yy >= np.min(y_reference) + distance) &
            (yy <= np.max(y_reference) - distance))

        z_valid = (
            (zz >= np.min(z_reference) + distance) &
            (zz <= np.max(z_reference) - distance))

        to_be_checked = (
            x_valid & y_valid & z_valid &
            dose_valid & gamma_valid
        )

        min_dose_diff = find_min_dose_difference_at_distance_3d(
            xx[to_be_checked], yy[to_be_checked], zz[to_be_checked],
            dose_evaluation[to_be_checked],
            reference_interpolation,
            distance, distance_step_size)

        gamma_at_distance = np.sqrt(
            min_dose_diff ** 2 / dose_threshold ** 2 +
            distance ** 2 / distance_threshold ** 2)

        running_gamma[to_be_checked] = np.min(
            np.vstack((
                    gamma_at_distance, running_gamma[to_be_checked]
                )), axis=0)

        gamma_valid = running_gamma > distance / distance_threshold

        distance += distance_step_size

        if (np.sum(to_be_checked) == 0) | (distance > maximum_test_distance):
            break

    return running_gamma
