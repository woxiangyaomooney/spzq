import cv2
import os

def split_video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    
    cap.release()
    print(f"视频已切割为 {frame_idx} 帧，并保存到文件夹: {output_folder}")


if __name__ == "__main__":
    # 分割视频为帧
    video_path = "data/tryy.mp4"
    output_folder = "data/frames1/"
    split_video_to_frames(video_path, output_folder)