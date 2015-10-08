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


def calc_gamma_2d(x, y, dose_evaluation, dose_reference,
                  distance_threshold=None, dose_threshold=None,
                  max_test_dist=None, step_group_size=24):
    if max_test_dist is None:
        max_test_dist = distance_threshold * 2

    x = np.array(x); y = np.array(y)
    dose_evaluation = np.array(dose_evaluation)
    dose_reference = np.array(dose_reference)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    assert np.all(np.abs(np.linspace(x[0], x[-1], len(x)) - x) < 0.0001)
    assert np.all(np.abs(np.linspace(y[0], y[-1], len(y)) - y) < 0.0001)

    mid_x_index = len(x) // 2
    mid_y_index = len(y) // 2

    xx_, yy_ = np.meshgrid(x, y)
    xx = np.ravel(xx_)
    yy = np.ravel(yy_)

    distance_from_middle = np.sqrt(
    (xx - x[mid_x_index])**2 +
    (yy - y[mid_y_index])**2)

    cut_off_dist = distance_from_middle < max_test_dist
    dist_sort_index = np.argsort(distance_from_middle[cut_off_dist])
    dist_sort = distance_from_middle[cut_off_dist][dist_sort_index]

    x_index = np.arange(len(x))
    y_index = np.arange(len(y))

    xx_index_, yy_index_ = np.meshgrid(x_index, y_index)
    xx_index = np.ravel(xx_index_)
    yy_index = np.ravel(yy_index_)

    x_dist_sort = (xx_index[cut_off_dist][dist_sort_index] - mid_x_index)
    y_dist_sort = (yy_index[cut_off_dist][dist_sort_index] - mid_y_index)

    i_start = y_dist_sort - np.min(y_dist_sort)
    i_start = i_start.tolist()

    i_end = y_dist_sort - np.max(y_dist_sort)
    ref = np.where(i_end == 0)[0]
    i_end = i_end.tolist()
    for i in ref:
        i_end[i] = None

    j_start = x_dist_sort - np.min(x_dist_sort)
    j_start = j_start.tolist()

    j_end = x_dist_sort - np.max(x_dist_sort)
    ref = np.where(j_end == 0)[0]
    j_end = j_end.tolist()
    for i in ref:
        j_end[i] = None

    x_new = x[j_start[0]:j_end[0]]
    y_new = y[i_start[0]:i_end[0]]

    running_minima = np.inf * np.ones_like(
        dose_evaluation[
            i_start[0]:i_end[0],
            j_start[0]:j_end[0]])

    for current_step in range(0, len(i_start), step_group_size):

        current_step_end = np.min([
            current_step + step_group_size,
            len(i_start)])
        current_test_set = range(
            current_step, current_step_end)

        dose_diff = np.array([
            (
                dose_evaluation[
                    i_start[i]:i_end[i],
                    j_start[i]:j_end[i]] -
                dose_reference[
                    i_start[i]:i_end[i],
                    j_start[i]:j_end[i]]
            )
         for i in current_test_set])

        distance = dist_sort[current_test_set][:, None, None]

        general_gamma = np.sqrt(
            dose_diff ** 2 / dose_threshold**2 +
            distance ** 2 / distance_threshold
        )

        running_general_gamma = np.vstack([
            general_gamma, running_minima[None, :, :]])

        running_minima = np.min(running_general_gamma, axis=0)

        stop = (
            np.max(running_minima) <
            np.sqrt(np.max(distance) ** 2 / distance_threshold ** 2))

        if stop:
            break

    gamma = running_minima

    return x_new, y_new, gamma


def calc_gamma_3d(x, y, z, dose_evaluation, dose_reference,
                  distance_threshold=None, dose_threshold=None,
                  max_test_dist=None, step_group_size=24):
    if max_test_dist is None:
        max_test_dist = distance_threshold * 2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    assert np.all(np.abs(np.linspace(x[0], x[-1], len(x)) - x) < 0.0001)
    assert np.all(np.abs(np.linspace(y[0], y[-1], len(y)) - y) < 0.0001)
    assert np.all(np.abs(np.linspace(z[0], z[-1], len(z)) - z) < 0.0001)

    mid_x_index = len(x) // 2
    mid_y_index = len(y) // 2
    mid_z_index = len(z) // 2

    xx_, yy_, zz_ = np.meshgrid(x, y, z)
    xx = np.ravel(xx_)
    yy = np.ravel(yy_)
    zz = np.ravel(zz_)

    distance_from_middle = np.sqrt(
        (xx - x[mid_x_index])**2 +
        (yy - y[mid_y_index])**2 +
        (zz - z[mid_z_index])**2
    )

    cut_off_dist = distance_from_middle < max_test_dist
    dist_sort_index = np.argsort(distance_from_middle[cut_off_dist])
    dist_sort = distance_from_middle[cut_off_dist][dist_sort_index]

    x_index = np.arange(len(x))
    y_index = np.arange(len(y))
    z_index = np.arange(len(z))

    xx_index_, yy_index_, zz_index_ = np.meshgrid(
        x_index, y_index, z_index)
    xx_index = np.ravel(xx_index_)
    yy_index = np.ravel(yy_index_)
    zz_index = np.ravel(zz_index_)

    x_dist_sort = (xx_index[cut_off_dist][dist_sort_index] - mid_x_index)
    y_dist_sort = (yy_index[cut_off_dist][dist_sort_index] - mid_y_index)
    z_dist_sort = (zz_index[cut_off_dist][dist_sort_index] - mid_z_index)

    i_start = y_dist_sort - np.min(y_dist_sort)
    i_start = i_start.tolist()
    i_end = y_dist_sort - np.max(y_dist_sort)
    ref = np.where(i_end == 0)[0]
    i_end = i_end.tolist()
    for i in ref:
        i_end[i] = None

    j_start = x_dist_sort - np.min(x_dist_sort)
    j_start = j_start.tolist()
    j_end = x_dist_sort - np.max(x_dist_sort)
    ref = np.where(j_end == 0)[0]
    j_end = j_end.tolist()
    for i in ref:
        j_end[i] = None

    k_start = z_dist_sort - np.min(z_dist_sort)
    k_start = k_start.tolist()
    k_end = z_dist_sort - np.max(z_dist_sort)
    ref = np.where(k_end == 0)[0]
    k_end = k_end.tolist()
    for i in ref:
        k_end[i] = None

    x_new = x[j_start[0]:j_end[0]]
    y_new = y[i_start[0]:i_end[0]]
    z_new = z[k_start[0]:k_end[0]]

    running_minima = np.inf * np.ones_like(
        dose_evaluation[
            i_start[0]:i_end[0],
            j_start[0]:j_end[0],
            k_start[0]:k_end[0]])

    for current_step in range(0, len(i_start), step_group_size):

        current_step_end = np.min([
            current_step + step_group_size,
            len(i_start)])
        current_test_set = range(
            current_step, current_step_end)

        dose_diff = np.array([
            (
                dose_evaluation[
                    i_start[i]:i_end[i],
                    j_start[i]:j_end[i],
                    k_start[i]:k_end[i]] -
                dose_reference[
                    i_start[i]:i_end[i],
                    j_start[i]:j_end[i],
                    k_start[i]:k_end[i]]
            )
         for i in current_test_set])

        distance = dist_sort[current_test_set][:, None, None, None]

        general_gamma = np.sqrt(
            dose_diff ** 2 / dose_threshold**2 +
            distance ** 2 / distance_threshold
        )

        running_general_gamma = np.vstack([
            general_gamma, running_minima[None, :, :, :]])

        running_minima = np.min(running_general_gamma, axis=0)

        stop = (
            np.max(running_minima) <
            np.sqrt(np.max(distance) ** 2 / distance_threshold ** 2))

        if stop:
            break

    gamma = running_minima

    return x_new, y_new, z_new, gamma
