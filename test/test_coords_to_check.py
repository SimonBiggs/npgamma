# Copyright (C) 2015-2016 Simon Biggs
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
from npgamma.main import coords_to_check_3d


def test_coords_stepsize():
    """Confirm that the the largest distance between one point and any other
    is less than the defined step size
    """
    distance = 1
    step_size = 0.1
    
    x, y, z = coords_to_check_3d(distance, step_size)
    
    distance_between_coords = np.sqrt(
        (x[:, None] - x[None, :])**2 + 
        (y[:, None] - y[None, :])**2 + 
        (z[:, None] - z[None, :])**2)
        
    largest_difference = np.max(np.min(distance_between_coords, axis=0))
    
    assert largest_difference <= step_size
