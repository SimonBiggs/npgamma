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


import yaml
import numpy as np
from npgamma.main import calc_gamma
from npgamma.main import GammaCalculation


class TestGamma():
    def setUp(self):
        grid = np.arange(0, 1, 0.1)
        self.dimensions = (len(grid), len(grid), len(grid))
        self.coords = (grid, grid, grid)
        self.reference = np.zeros(self.dimensions)
        self.reference[2:-2:, 2:-2:, 2:-2:] = 1

        self.evaluation = np.zeros(self.dimensions)
        self.evaluation[3:-2:, 4:-2:, 5:-2:] = 1.015
        
        self.expected_gamma = np.zeros(self.dimensions)
        self.expected_gamma[2:-2:, 2:-2:, 2:-2:] = 0.4
        self.expected_gamma[3:-3:, 3:-3:, 3:-3:] = 0.7
        self.expected_gamma[4:-4:, 4:-4:, 4:-4:] = 1
        self.expected_gamma[3:-2:, 4:-2:, 5:-2:] = 0.5
        
        
    def test_regression_of_gamma_3d(self):    
        self.gamma3d = np.round(calc_gamma(
            self.coords, self.reference,
            self.coords, self.evaluation,
            0.3, 0.03), decimals=3)
            
        assert np.all(self.expected_gamma == self.gamma3d)
            
            
    def test_regression_of_gamma_2d(self):    
        self.gamma2d = np.round(calc_gamma(
            self.coords[0:2], self.reference[5,:,:],
            self.coords[0:2], self.evaluation[5,:,:],
            0.3, 0.03), decimals=3)
            
        assert np.all(self.expected_gamma[5,:,:] == self.gamma2d)
        
    
    def test_regression_of_gamma_1d(self):    
        self.gamma1d = np.round(calc_gamma(
            self.coords[0], self.reference[5,5,:],
            self.coords[0], self.evaluation[5,5,:],
            0.3, 0.03), decimals=3)
            
        assert np.all(self.expected_gamma[5,5,:] == self.gamma1d)
        
        
    def test_coords_stepsize(self):
        """Confirm that the the largest distance between one point and any other
        is less than the defined step size
        """        
        gamma_calculation = GammaCalculation(
            self.coords, self.reference,
            self.coords, self.evaluation,
            0.3, 0.03)
            
        y, x, z = gamma_calculation.calculate_coordinates_kernel(1)
        
        distance_between_coords = np.sqrt(
            (x[:, None] - x[None, :])**2 + 
            (y[:, None] - y[None, :])**2 + 
            (z[:, None] - z[None, :])**2)

        distance_between_coords[
          distance_between_coords == 0] = np.nan
            
        largest_difference = np.max(np.nanmin(distance_between_coords, axis=0))
        
        assert largest_difference <= gamma_calculation.distance_step_size
        assert largest_difference > gamma_calculation.distance_step_size * 0.9
        
        
