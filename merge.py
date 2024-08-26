import os
import cv2

def combine_frames_to_video(image_folder, output_video_path, fps):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if not images:
        print("未找到任何图像")
        return

    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"增强视频已保存为: {output_video_path}")

if __name__ == "__main__":
    # 将帧合并为视频
    image_folder = "results/dl/frames/"
    output_video_path = "results/try_enhanced_video.mp4"
    fps = 25
    combine_frames_to_video(image_folder, output_video_path, fps)