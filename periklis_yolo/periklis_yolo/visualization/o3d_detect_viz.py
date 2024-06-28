import open3d as o3d
import numpy as np

class Open3DDetectVisualizer:
    def __init__(self, name='Open3D'):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(name, width=640, height=480)
        self.setup_visualizer()

    def setup_visualizer(self):
        # Add 8 points to initiate the visualizer's bounding box
        points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [10, 0, 0],
            [10, 0, 1],
            [10, 1, 0],
            [10, 1, 1]
        ])

        points *= 4

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(pcd, reset_bounding_box=True)

        view_control = self.vis.get_view_control()
        view_control.rotate(0, -525)
        view_control.rotate(500, 0)

        # points thinner and lines thicker
        self.vis.get_render_option().point_size = 2.0
        self.vis.get_render_option().line_width = 10.0

    def reset(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5), reset_bounding_box=False)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_bbox(self, bbox3d_points, color=[1, 0, 0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bbox3d_points)
        self.vis.add_geometry(pcd, reset_bounding_box=False)
        pcd.paint_uniform_color(color)

        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = color
        self.vis.add_geometry(bbox, reset_bounding_box=False)

    def draw_pointcloud(self, pc2_points_64, transformation_matrix=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc2_points_64)

        if transformation_matrix is not None:
            pcd.transform(transformation_matrix)

        self.vis.add_geometry(pcd, reset_bounding_box=False)