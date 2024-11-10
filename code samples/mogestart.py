import cv2
import torch
from moge.model import MoGeModel

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)                             

# Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
input_image = cv2.cvtColor(cv2.imread("example_images/MaitreyaBuddha.png"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

print("start inferencing")
# Infer 
output = model.infer(input_image)
# `output` has keys "points", "depth", "mask" and "intrinsics",
# The maps are in the same size as the input image. 
# {
#     "points": (H, W, 3),    # scale-invariant point map in OpenCV camera coordinate system (x right, y down, z forward)
#     "depth": (H, W),        # scale-invariant depth map
#     "mask": (H, W),         # a binary mask for valid pixels. 
#     "intrinsics": (3, 3),   # normalized camera intrinsics
# }

#plot depth 
depth = output["depth"].clone()  # Make a copy of the tensor
max_finite_value = torch.max(depth[torch.isfinite(depth)])
depth[torch.isinf(depth)] = max_finite_value  # Replace inf values with the maximum finite value
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = (depth.cpu().numpy() * 255).astype("uint8")  # Normalize to 0-255 and convert to uint8

# Apply inferno colormap using OpenCV
depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

import numpy as np
input_image = cv2.imread("example_images/MaitreyaBuddha.png")
input_image = np.array(input_image)
depth_colored =  np.array(depth_colored)

# Display the input image and depth map
result = np.hstack((input_image, depth_colored))
result = cv2.resize(result, (int(result.shape[1] // 4 * 3), int(result.shape[0] // 4 * 3)))
cv2.imshow("Input Image vs Depth Map", result)
cv2.waitKey(0)

cv2.destroyAllWindows()
print(output)

import open3d as o3d
# Convert the points to a point cloud
points = output["points"].clone()
points = points[torch.isfinite(points).all(dim=-1)]  # Remove NaN values
points = points.cpu().numpy()

# Create a point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
np.savetxt("point_cloud.txt", np.vstack(points), fmt='%.6f')
# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])



