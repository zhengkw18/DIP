import argparse
import json
import cv2
from scipy.spatial import Delaunay


class Delaunay:
    """
    A 2D Delaunay triangulation solver using naive Bowyer-Watson algorithm
    """

    def __init__(self):
        self.points = [[0, 1e6], [-1e6, -1e6], [1e6, -1e6]]
        self.triangles = {}
        self.circles = {}
        self.triangles[(0, 1, 2)] = [None, None, None]
        self.circles[(0, 1, 2)] = self.cal_circum((0, 1, 2))

    def put(self, point):
        def _in_circle(tri, p):
            o = self.circles[tri]
            return (p[0] - o[0]) ** 2 + (p[1] - o[1]) ** 2 <= o[2]

        idx = len(self.points)
        self.points.append(point)
        bad_tris = []
        for tri in self.triangles:
            if _in_circle(tri, point):
                bad_tris.append(tri)
        bound, tri, edge = [], bad_tris[0], 0
        while True:
            tri_op = self.triangles[tri][edge]
            if tri_op in bad_tris:
                edge = (self.triangles[tri_op].index(tri) + 1) % 3
                tri = tri_op
            else:
                bound.append((tri[(edge + 1) % 3], tri[(edge - 1) % 3], tri_op))
                edge = (edge + 1) % 3
                if bound[0][0] == bound[-1][1]:
                    break
        for tri in bad_tris:
            self.triangles.pop(tri)
            self.circles.pop(tri)
        new_tris = []
        for e0, e1, tri_op in bound:
            new_tri = (idx, e0, e1)
            new_tris.append(new_tri)
            self.triangles[new_tri] = [tri_op, None, None]
            self.circles[new_tri] = self.cal_circum(new_tri)
            if tri_op:
                for i, tri_op_op in enumerate(self.triangles[tri_op]):
                    if tri_op_op and e0 in tri_op_op and e1 in tri_op_op:
                        self.triangles[tri_op][i] = new_tri
        for i, new_tri in enumerate(new_tris):
            self.triangles[new_tri][1], self.triangles[new_tri][2] = new_tris[(i + 1) % len(new_tris)], new_tris[(i - 1) % len(new_tris)]

    def cal_circum(self, triangle):
        [ax, ay], [bx, by], [cx, cy] = self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        r2 = (ax - ux) ** 2 + (ay - uy) ** 2
        return ux, uy, r2

    def get_triangles(self):
        return [(a - 3, b - 3, c - 3) for (a, b, c) in self.triangles if a >= 3 and b >= 3 and c >= 3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Delaunay triangulation for an image with keypoints and output diagrams")
    parser.add_argument("--src", type=str, help="Source image", required=True)

    args = parser.parse_args()
    with open(args.src + ".json", "r") as f:
        lst = json.load(f)
    src_img = cv2.imread(args.src)
    H, W, _ = src_img.shape
    lst.extend([[0, 0], [0, W // 2], [0, W - 1], [H // 2, 0], [H // 2, W - 1], [H - 1, 0], [H - 1, W // 2], [H - 1, W - 1]])
    # d = Delaunay(lst)
    # tri = d.simplices.astype("int")
    # triangles = []
    # for i in range(len(tri)):
    #     triangles.append([int(tri[i][0]), int(tri[i][1]), int(tri[i][2])])
    d = Delaunay()
    for e in lst:
        d.put(e)
    triangles = d.get_triangles()
    with open(args.src + ".tris.json", "w") as f:
        json.dump(triangles, f)
    src_points, src_tris = src_img.copy(), src_img.copy()
    for point in lst:
        cv2.circle(src_points, tuple(reversed(point)), 1, (0, 0, 255))
    for tri in triangles:
        cv2.line(src_tris, tuple(reversed(lst[tri[0]])), tuple(reversed(lst[tri[1]])), (0, 0, 255))
        cv2.line(src_tris, tuple(reversed(lst[tri[1]])), tuple(reversed(lst[tri[2]])), (0, 0, 255))
        cv2.line(src_tris, tuple(reversed(lst[tri[2]])), tuple(reversed(lst[tri[0]])), (0, 0, 255))
    cv2.imwrite(args.src.split(".")[0] + "_points.jpg", src_points)
    cv2.imwrite(args.src.split(".")[0] + "_tris.jpg", src_tris)
