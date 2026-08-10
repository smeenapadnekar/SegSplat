# import open3d as o3d
import numpy as np

# ply_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/fern/point_cloud/iteration_5000/point_cloud.ply"

# pcd = o3d.io.read_point_cloud(ply_file)
# points = np.asarray(pcd.points)
# geoms = []

# for pt in points:
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#     sphere.translate(pt)
#     geoms.append(sphere)
# print("tadaa")
# o3d.visualization.draw_geometries(geoms)



import open3d as o3d
from open3d.web_visualizer import draw


ply_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/counter/point_cloud/iteration_10000/point_cloud.ply"
# ply_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/fern/point_cloud/iteration_10000/point_cloud.ply"
ply_3dgs = o3d.t.io.read_point_cloud(ply_file)
print(ply_3dgs)
pcd = o3d.io.read_point_cloud(ply_file)
colors = np.asarray(pcd.colors)
print(colors.shape)  # Should be (N, 3) if colors are present

if colors.shape[0] == 0:
    print("No color loaded from PLY!")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024]
                                  )


# draw(ply_3dgs)
