import torch
import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equal(self, q):
        if q.x == self.x and q.y == self.y:
            return True
        else:
            return False


# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # exception
    if (p1.equal(p2) or p1.equal(q2) or q1.equal(p2) or q1.equal(q2)) and ((o1 != o2) and (o3 != o4)):
        return False

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


# Driver program to test above functions:
# p1 = Point(.1, .3)
# q1 = Point(.4, .9)
# p2 = Point(.31, .7)
# q2 = Point(.6, .2)
#
#
# # plt.axline((p1.x, p1.y), (q1.x, q1.y),  color='k', linestyle='-', linewidth=2)
# # plt.axline((p2.x, p2.y), (q2.x, q2.y),  color='b', linestyle='-', linewidth=2)
# plt.plot([p1.x, q1.x], [p1.y, q1.y],  color='b', linestyle='-', linewidth=2)
# plt.plot([p2.x, q2.x], [p2.y, q2.y],  color='k', linestyle='-', linewidth=2)
# plt.show()
#
# if doIntersect(p1, q1, p2, q2):
#     print("Yes")
# else:
#     print("No")

if __name__ == "__main__":
    #
    # p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

    # p1 = Point(0.2, 0.2)
    # q1 = Point(0.7, 0.2)
    # p2 = Point(0.2, 0.2)
    # q2 = Point(0.5, 0.2)

    # node = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # p1 = p2 = node
    # q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

    # node = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # p1 = q2 = node
    # p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

    # node = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q1 = p2 = node
    # p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    # q2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))

    node = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    q1 = q2 = node
    p1 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))
    p2 = Point(np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0))



    indices = np.random.randint(low=0, high=4, size=2)

    if doIntersect(p1, q1, p2, q2):
        print("Yes")
    else:
        print("No")
    plt.plot([p1.x, q1.x], [p1.y, q1.y], color='b', linestyle='-', linewidth=2)
    plt.plot([p2.x, q2.x], [p2.y, q2.y], color='k', linestyle='-', linewidth=2)
    plt.show()
