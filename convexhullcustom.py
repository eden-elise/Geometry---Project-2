import numpy as np

from numpy.typing import NDArray







class ConvexHull_C():
    def __init__(self, points):
      self.points = points
      self.hull, self.vertices = self.convex_hull_custom(points)

    def convex_hull_custom(self, points: np.ndarray) -> tuple[NDArray, NDArray]:
      start_idx = np.argmin(points[:, 1])
      angles = np.arctan2(points[:, 1] - points[start_idx, 1], points[:, 0] - points[start_idx, 0])
      points_sorted = points[np.argsort(angles)]
      hull = [points[start_idx], points_sorted[1]]
      for i in range(len(points_sorted)-1):
        while len(hull) > 1 and np.cross(hull[-1] - hull[-2], points_sorted[i+1] - hull[-2]) <= 0:
          hull.pop()
        hull.append(points_sorted[i+1]) #Actual points,
      
        vertices = []
        for h in hull:
          idx = np.where((points == h).all(axis=1))[0][0]
          vertices.append(idx) #Indexes of points

  
      return np.array(hull), np.array(vertices) 
    
#TEST CODE BELOW
    
#import matplotlib.pyplot as plt

# np.random.seed(3)
# points = np.random.rand(10, 2)


# stard_idx = np.argmin(points[:, 1])
# angles = np.arctan2(points[:, 1] - points[stard_idx, 1], points[:, 0] - points[stard_idx, 0])
# sorted_indices = np.argsort(angles)
# points1 = points[sorted_indices]

# hull = [points[stard_idx],points1[1]]

# for i in range(len(points1)-1):
#   while len(hull) > 1 and np.cross(hull[-1] - hull[-2], points1[i+1] - hull[-2]) <= 0:
#     hull.pop()
#   hull.append(points1[i+1])

# hull = np.array(hull)
# print(points[stard_idx])
# print(len(hull))




#   stard_idx = np.argmin(points[:, 1])
#   angles = np.arctan2(points[:, 1] - points[stard_idx, 1], points[:, 0] - points[stard_idx, 0])
#   sorted_indices = np.argsort(angles)
#   points1 = points[sorted_indices]

#   hull = [points[stard_idx],points1[1]]

#   for i in range(len(points1)-1):
#     while len(hull) > 1 and np.cross(hull[-1] - hull[-2], points1[i+1] - hull[-2]) <= 0:
#       hull.pop()
#     hull.append(points1[i+1])

#   return np.array(hull)


# if True:
#   plt.scatter(points[:, 0], points[:, 1], s=50, color='blue')
#   plt.scatter(hull[:, 0], hull[:, 1], s=50, color='red')
#   plt.title('Random Points')
#   plt.xlabel('X-axis')
#   plt.ylabel('Y-axis')
#   plt.xlim(0, 1)
#   plt.ylim(0, 1)
#   plt.grid()
#   plt.show()