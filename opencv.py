#problem1
"""import cv2
import numpy as np
import pyrealsense2 as rs  # If using Intel RealSense

def get_distance(depth_frame, x, y):
    """Get the distance from the depth frame at the specified (x, y) coordinates."""
    distance = depth_frame.get_distance(x, y)
    return distance

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Start streaming
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting the pipeline: {e}")
        return

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                print("Could not retrieve frames.")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Get the distance at the center of the frame
            height, width, _ = color_image.shape
            center_x, center_y = width // 2, height // 2
            distance = get_distance(depth_frame, center_x, center_y)

            # Check if the distance is valid
            if distance < 0:  # Assuming valid distance should not be negative
                distance_text = "Distance: N/A"
            else:
                distance_text = f"Distance: {distance:.2f} meters"

            # Display the distance on the image
            cv2.putText(color_image, distance_text, (center_x, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show images
            cv2.imshow('Depth Image', depth_image)
            cv2.imshow('Color Image', color_image)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    """

#problem2
"""
import cv2
import torch

# Load YOLOv5 model (we'll use the pre-trained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can change the model to 'yolov5m', 'yolov5l', or 'yolov5x'

# Define video input
video_path = 'your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Perform inference on the current frame
    results = model(frame)

    # Render the results on the frame
    results.render()  # Render the boxes on the frame
    
    # Display the frame with the detections
    cv2.imshow("Frame", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
"""


#Problem3
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from midas.dpt_depth import DPTDepthModel
from midas.utils import MidasNet, MidasNetSmall, resize, normalize, colormap

# Load the MiDaS model (Monocular Depth Estimation)
model_type = "DPT_Large"  # Use a large model for better accuracy
model = DPTDepthModel(model_type)
model.eval()

# Load the pre-trained weights
checkpoint_path = "dpt_large-5d3b4c8f.pt"  # Download the pre-trained model from MiDaS GitHub
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model"])

# Video Input
video_path = "your_video.mp4"  # Path to the input MP4 video
cap = cv2.VideoCapture(video_path)

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to process each frame and create depth map
def get_depth_map(frame):
    # Resize image to fit the model input
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = resize(input_image, 384, 384)
    
    # Normalize the image to the range [0, 1]
    input_image = normalize(input_image)

    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()

    # Run inference
    with torch.no_grad():
        depth_map = model(input_tensor)
        
    # Process the depth map
    depth_map = depth_map.squeeze().cpu().numpy()

    # Scale the depth map to the range [0, 255] for visualization
    depth_map_normalized = np.uint8(depth_map / depth_map.max() * 255)

    return depth_map_normalized

# Navigation system based on depth
def navigation_system(depth_map):
    # In a real system, we'd use the depth map to detect obstacles
    # For simplicity, let's consider a basic navigation rule:
    # - If an obstacle is close (depth < threshold), stop or change direction.
    
    # Find the minimum depth in the center region of the depth map
    height, width = depth_map.shape
    center_depth = depth_map[height//3:2*height//3, width//3:2*width//3].min()

    # Define a threshold for obstacle detection (closer objects)
    threshold = 50  # You can adjust this value depending on the depth range

    # If the minimum depth is less than the threshold, an obstacle is detected
    if center_depth < threshold:
        print("Obstacle detected! Adjusting navigation...")
        # Implement navigation changes here, such as steering or stopping
    else:
        print("Path is clear, continuing navigation...")

# Process the video and generate depth maps
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # End of video
    
    # Generate depth map for the current frame
    depth_map = get_depth_map(frame)

    # Display the depth map
    cv2.imshow("Depth Map", depth_map)

    # Implement basic navigation system
    navigation_system(depth_map)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

"""