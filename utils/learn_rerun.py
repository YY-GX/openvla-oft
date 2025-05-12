# import rerun as rr
# import numpy as np
#
# rr.init("rerun_example_my_data", spawn=True)
#
# SIZE = 10
#
# pos_grid = np.meshgrid(*[np.linspace(-10, 10, SIZE)]*3)
# positions = np.vstack([d.reshape(-1) for d in pos_grid]).T
#
# col_grid = np.meshgrid(*[np.linspace(0, 255, SIZE)]*3)
# colors = np.vstack([c.reshape(-1) for c in col_grid]).astype(np.uint8).T
#
# rr.log(
#     "my_points",
#     rr.Points3D(positions, colors=colors, radii=0.5)
# )

import rerun as rr

from math import tau
import numpy as np
from rerun.utilities import build_color_spiral
from rerun.utilities import bounce_lerp

rr.init("rerun_example_dna_abacus", spawn=True)

NUM_POINTS = 100

# Points and colors are both np.array((NUM_POINTS, 3))
points1, colors1 = build_color_spiral(NUM_POINTS)
points2, colors2 = build_color_spiral(NUM_POINTS, angular_offset=tau*0.5)

rr.log("dna/structure/left", rr.Points3D(points1, colors=colors1, radii=0.08))
rr.log("dna/structure/right", rr.Points3D(points2, colors=colors2, radii=0.08))

