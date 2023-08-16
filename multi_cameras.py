from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from PIL import Image
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.usd import UsdGeom
from pxr import UsdGeom

import os
import json
import omni.usd
import numpy as np
import open3d as o3d
import omni.isaac.range_sensor
import omni.replicator.core as rep

"""
    Geting Render Product Data from multi cameras.
"""

# Function to create a camera in isaac-sim
def CreateCamera (
        CameraName: str="/camera_link",
        Projection: str=UsdGeom.Tokens.perspective,
        FocalLength: float=20,
        HorizontalAperture: float=20.955,
        VerticalAperture: float=15.2908,
        ClippingRange: tuple=(1, 100000),
        TranslateOp: tuple=(0., 0., 0.),
        RotateZYXOp: tuple=(0., 0., 0.)
    ) -> UsdGeom.Camera:
    stage = omni.usd.get_context().get_stage()
    usd_camera = UsdGeom.Camera.Define(stage, CameraName)
    usd_camera.CreateProjectionAttr().Set(Projection)
    usd_camera.CreateFocalLengthAttr().Set(FocalLength)
    # Set a few other common attributes too
    usd_camera.CreateHorizontalApertureAttr().Set(HorizontalAperture)
    usd_camera.CreateVerticalApertureAttr().Set(VerticalAperture)
    usd_camera.CreateClippingRangeAttr().Set(ClippingRange)
    usd_camera.AddTranslateOp().Set(TranslateOp)
    usd_camera.AddRotateZYXOp().Set(RotateZYXOp)
    return usd_camera

# Util function to save rgb annotator data
def write_rgb_data(rgb_annots: list, file_path: str) -> None:
    os.makedirs(file_path, exist_ok=True)
    for i, rgb_annot in enumerate(rgb_annots):
        # create camera file
        rgb_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(rgb_camera_path, exist_ok=True)
        rgb_data = rgb_annot.get_data()
        # save numpy data
        np.save(os.path.join(rgb_camera_path, "rgb.npy"), rgb_data)
        # save RGBA image
        rgb_image_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape, -1)
        rgb_img = Image.fromarray(rgb_image_data, "RGBA")
        rgb_img.save(os.path.join(rgb_camera_path, "rgb.png"))

# Util function to save normals annotator data
def write_nor_data(nor_annots: list, file_path: str) -> None:
    os.makedirs(file_path, exist_ok=True)
    for i, nor_annot in enumerate(nor_annots):
        # create camera file
        nor_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(nor_camera_path, exist_ok=True)
        nor_data = nor_annot.get_data()
        # save numpy data
        np.save(os.path.join(nor_camera_path, "normals.npy"), nor_data)

# Util function to save pcd annotator data
def write_pcd_data(pcd_annots: list, file_path: str) -> None:
    os.makedirs(file_path, exist_ok=True)
    for i, pcd_annot in enumerate(pcd_annots):
        # create camera file
        pcd_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(pcd_camera_path, exist_ok=True)
        pcd_data = pcd_annot.get_data()
        # save numpy data
        points = pcd_data["data"]
        normals = pcd_data["info"]["pointNormals"].reshape(-1, 4)[:, 0:3]
        colors = pcd_data["info"]["pointRgb"].reshape(-1, 4)[:, 0:3] / 255.0
        np.save(os.path.join(pcd_camera_path, "pointcloud.npy"), points)
        np.save(os.path.join(pcd_camera_path, "pointcloud_normals.npy"), normals)
        np.save(os.path.join(pcd_camera_path, "pointcloud_rgb.npy"), colors)
        # save synthetic pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(os.path.join(pcd_camera_path, "synthetic_pointcloud.pcd"), pcd)

# Util function to save camera params annotator data
def write_cam_data(cam_annots: list, file_path: str) -> None:
    os.makedirs(file_path, exist_ok=True)
    for i, cam_annot in enumerate(cam_annots):
        # create camera file
        pram_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(pram_camera_path, exist_ok=True)
        cam_data = cam_annot.get_data()
        # save camera parameters
        dict2json(os.path.join(pram_camera_path, "camera_params.json"), cam_data)

# Util function to save distance to camera annotator data
def write_d2c_data(d2c_annots: list, file_path: str) -> None:
    """
    Outputs a depth map from objects to camera positions. The annotator produces a 2d array of types with 1 channel.
    Note:
        1. The unit for distance to camera is in meters.
        2. 0 in the 2d array represents infinity (which means there is no object in that pixel).
    """
    os.makedirs(file_path, exist_ok=True)
    for i, d2c_annot in enumerate(d2c_annots):
        # create camera file
        d2c_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(d2c_camera_path, exist_ok=True)
        d2c_data = d2c_annot.get_data()
        # save numpy data
        np.save(os.path.join(d2c_camera_path, "distance_to_camera.npy"), d2c_data)

# Util function to save distance to image plane annotator data
def write_d2i_data(d2i_annots: list, file_path: str) -> None:
    """
    Outputs a depth map from objects to image plane of the camera. The annotator produces a 2d array of types with 1 channel.
    Note:
        1. The unit for distance to camera is in meters.
        2. 0 in the 2d array represents infinity (which means there is no object in that pixel).
    """
    os.makedirs(file_path, exist_ok=True)
    for i, d2i_annot in enumerate(d2i_annots):
        # create camera file
        d2i_camera_path = os.path.join(file_path, f"camera_{i + 1}")
        os.makedirs(d2i_camera_path, exist_ok=True)
        d2i_data = d2i_annot.get_data()
        # save numpy data
        np.save(os.path.join(file_path, "distance_to_image_plane.npy"), d2i_data)

# Function to save class 'dict' to .json
class NumpyArrayEncoder(json.JSONEncoder):
    """
    Python dictionaries can store ndarray array types, but when serialized by dict into JSON files, 
    ndarray types cannot be serialized. In order to read and write numpy arrays, 
    we need to rewrite JSONEncoder's default method. 
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dict2json(file_name: str, the_dict: dict) -> None: 
    try:
        json_str = json.dumps(the_dict, cls=NumpyArrayEncoder, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0   
    
# Accesing the data directly from annotators
def Get_Data_Annotator(rep_cameras: list):
    rgb_annots, nor_annots, pcd_annots, cam_annots, d2c_annots, d2i_annots, sem_annots, ins_annots, ins_id_annots = [], [], [], [], [], [], [], [], []
    for rep_camera in rep_cameras:
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb").attach([rep_camera]); rgb_annots.append(rgb_annot)
        nor_annot = rep.AnnotatorRegistry.get_annotator("normals").attach([rep_camera]); nor_annots.append(nor_annot)
        pcd_annot = rep.AnnotatorRegistry.get_annotator("pointcloud").attach([rep_camera]); pcd_annots.append(pcd_annot)
        cam_annot = rep.AnnotatorRegistry.get_annotator("CameraParams").attach(rep_camera); cam_annots.append(cam_annot)
        d2c_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera").attach([rep_camera]); d2c_annots.append(d2c_annot)
        d2i_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane").attach([rep_camera]); d2i_annots.append(d2i_annot)
        sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True}).attach([rep_camera]); sem_annots.append(sem_annot)
        ins_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": True}).attach([rep_camera]); ins_annots.append(ins_annot)
        ins_id_annot = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": True}).attach([rep_camera]); ins_id_annots.append(ins_id_annot)
    return rgb_annots, nor_annots, pcd_annots, cam_annots, d2c_annots, d2i_annots, sem_annots, ins_annots, ins_id_annots

if __name__=="__main__":
    # Load scene
    my_world = World()
    my_world.scene.add_default_ground_plane()
    scene_path = "main.usd" # House_22
    add_reference_to_stage(usd_path=scene_path, prim_path="/World/Scene")
    my_world.reset()

    # Run the application for several frames to allow the materials to load
    for i in range(20):
        simulation_app.update()

    # Create a camera
    usd_camera_1 = CreateCamera(CameraName="/Camera_link1", TranslateOp=(-8, -9, 0.8), RotateZYXOp=(90., 90., 0.))
    usd_camera_2 = CreateCamera(CameraName="/Camera_link2", TranslateOp=(-14, -8, 1.0), RotateZYXOp=(95., 160., 0.), FocalLength=10.0)
    
    # Create render products
    rep_cameras = []
    rep_camera_1 = rep.create.render_product(str(usd_camera_1.GetPath()), resolution=(640, 480)); rep_cameras.append(rep_camera_1)
    rep_camera_2 = rep.create.render_product(str(usd_camera_2.GetPath()), resolution=(640, 480)); rep_cameras.append(rep_camera_2)

    # Set the output directory for the data
    out_dir = os.getcwd() + "/out_sim_get_data"
    os.makedirs(out_dir, exist_ok=True)

    # Accesing the data directly from annotators
    rgb_annots, nor_annots, pcd_annots, cam_annots, d2c_annots, d2i_annots, sem_annots, ins_annots, ins_id_annots = Get_Data_Annotator(rep_cameras)

    # NOTE replicator's step is needed
    rep.orchestrator.step()

    # Save render product data
    write_rgb_data(rgb_annots, f"{out_dir}/rgb")
    write_nor_data(nor_annots, f"{out_dir}/normals")
    write_pcd_data(pcd_annots, f"{out_dir}/pointcloud")
    write_cam_data(cam_annots, f"{out_dir}/camera_params")
    write_d2c_data(d2c_annots, f"{out_dir}/distance_to_camera")
    write_d2i_data(d2i_annots, f"{out_dir}/distance_to_image_plane")

    # while simulation_app.is_running():
    #     simulation_app.update()
    simulation_app.close()
