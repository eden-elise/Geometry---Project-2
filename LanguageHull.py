"""
Geometry Project Two - Language Convex Hulls
Authors: Eden T, Luke F, Jacob R
Process: Word(string) --> TextPath --> Convex Hull

Each word is rendered as font geometry using matplotlib's TextPath. We then
compute the convex hull of those outlines and extract shape features.
Different scripts (Latin, Arabic, Chinese) are expected to produce geometrically 
distinct hulls.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.spatial import ConvexHull
