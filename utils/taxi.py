import numpy as np
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] # set root directory
sys.path.append(os.path.abspath(ROOT))

def curise_region(time, start_region, travelingdistance_region_to_region, num_of_regions=38, idle_dis_dt=2.5):
    potential_region = []
    for i in range(num_of_regions):
        if travelingdistance_region_to_region[start_region][i] < 2.5:
            potential_region.append(i)
    return np.random.choice(potential_region)




class Taxi():

    def __init__(self,location=0, status=0, destination=-1,arriving_time=0,remaining_energy=0) -> None:
        self.location = location
        self.status = status
        self.destination = destination
        self.arriving_time = arriving_time
        self.remaining_energy = remaining_energy
        self.charging_status = 0
        self.update_status = 1
        
if __name__ == '__main__':
    pass
