import os
import cv2


def vdo_to_frames(vdo_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    cap = cv2.VideoCapture(vdo_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {vdo_path}")
        return
    
    frame_idx = 0  # Frame index counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames are available

        # Generate filename based on frame index
        frame_filename = os.path.join(save_path, f"{frame_idx}.png")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)

        print(f"Saved: {frame_filename}")
        frame_idx += 1

    cap.release()
    print(f"Video split into {frame_idx} frames in {save_path}")


file_dir = os.path.dirname(os.path.abspath(__file__))

vdo_name = '2024-07-07_2024-07-07_17-00-03.mp4'
vdo_path = os.path.join(file_dir, 'data', vdo_name)
save_path = os.path.join(file_dir, 'data', 'frames', vdo_name.split('.')[0])

vdo_to_frames(vdo_path, save_path)

