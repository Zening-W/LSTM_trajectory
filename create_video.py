import cv2
import os


vdo_folder = 'model2_45'

dir_path = os.path.dirname(os.path.abspath(__file__))
frames_path = os.path.join(dir_path, 'test', vdo_folder)
output_path = os.path.join(dir_path, 'videos', vdo_folder+'.mp4')


# Parameters for the video
fps = 10  # Frames per second
frame_size = None  # Automatically determined by the first frame

# Get sorted list of frames
frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

if not frame_files:
    print("No frames found in the directory!")
    exit()

# Read the first frame to determine frame size
first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Write frames to video
for frame_file in frame_files:
    frame_path = os.path.join(frames_path, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Failed to read frame {frame_file}, skipping...")
        continue
    video_writer.write(frame)

# Release the VideoWriter
video_writer.release()
print(f"Video saved to {output_path}")
