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


def calc_gamma_unequal(x_reference, y_reference, dose_reference,
               x_evaluation, y_evaluation, dose_evaluation,
               dose_threshold=None, distance_threshold=None):

    xref, yref = np.meshgrid(x_reference, y_reference)
    xref = np.ravel(xref)
    yref = np.ravel(yref)
    dose_ref = np.ravel(dose_reference)

    xeval, yeval = np.meshgrid(x_evaluation, y_evaluation)
    xeval = np.ravel(xeval)
    yeval = np.ravel(yeval)
    dose_eval = np.ravel(dose_evaluation)

    distances = np.array([
        np.sqrt(
            (xeval[i] - xref)**2 +
            (yeval[i] - yref)**2
        )
        for i in range(len(xeval))
    ])

    dose_diff = np.array([
        dose_eval_i - dose_ref
        for dose_eval_i in dose_eval
    ])

    generalised_gamma = np.sqrt(
        distances**2 / distance_threshold**2 +
        dose_diff**2 / dose_threshold**2
    )

    gamma_before_reshape = np.min(generalised_gamma, axis=1)
    gamma = np.reshape(gamma_before_reshape, np.shape(dose_evaluation))

    return gamma
