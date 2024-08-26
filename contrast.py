import sys
import os
import cv2
import numpy as np
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QProgressBar, QCheckBox, QComboBox, QGridLayout, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Define paths for saving results
RESULTS_DIR = 'results'
TRADITIONAL_DIR = os.path.join(RESULTS_DIR, 'traditional')
DL_DIR = os.path.join(RESULTS_DIR, 'dl')

# Ensure result directories exist
os.makedirs(TRADITIONAL_DIR, exist_ok=True)
os.makedirs(DL_DIR, exist_ok=True)

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, input_path, is_video, traditional_methods, dl_model, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.is_video = is_video
        self.traditional_methods = traditional_methods
        self.dl_model = dl_model

    def run(self):
        if self.traditional_methods:
            if self.is_video:
                self.process_video_traditional()
            else:
                self.process_images_traditional()
        if self.is_video:
            self.process_video_dl()
        else:
            self.process_images_dl()
        self.finished.emit()

    def process_images_traditional(self):
        files = [f for f in os.listdir(self.input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(files)
        for i, file_name in enumerate(files):
            image_path = os.path.join(self.input_path, file_name)
            image = cv2.imread(image_path)
            image = enhance_traditional(image)
            output_path = os.path.join(TRADITIONAL_DIR, file_name)
            cv2.imwrite(output_path, image)
            self.progress.emit(int(100 * (i + 1) / total_files))

    def process_video_traditional(self):
        cap = cv2.VideoCapture(self.input_path)
        
        # 获取输入视频的帧率和分辨率
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 提取输入视频的文件名，并保留扩展名
        input_filename = os.path.basename(self.input_path)
        output_path = os.path.join(TRADITIONAL_DIR, input_filename)
        
        # 使用与输入视频匹配的编码器和参数创建VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = enhance_traditional(frame)
            out.write(frame)
            frame_idx += 1
            self.progress.emit(int(100 * frame_idx / total_frames))
        
        cap.release()
        out.release()


    def process_images_dl(self):
        command = [
            'python', 'inference_realbasicvsr.py',
            'configs/realbasicvsr_x4.py',
            f'checkpoints/{self.dl_model}.pth',
            self.input_path,
            DL_DIR
        ]
        subprocess.run(command)
        
    def process_video_dl(self):
         # 提取输入视频的文件名，并保留扩展名
        input_filename = os.path.basename(self.input_path)
        command = [
            'python', 'inference_realbasicvsr.py',
            'configs/realbasicvsr_x4.py',
            f'checkpoints/{self.dl_model}.pth',
            self.input_path,
            os.path.join(DL_DIR, input_filename),
            '--fps', '25',
            '--max_seq_len', '2'
        ]
        subprocess.run(command)

def enhance_traditional(image):
    image = sharpen_image(image)
    image = filter_image(image)
    image = interpolate_image(image)
    return image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def filter_image(image):
    # Apply a basic filter, e.g., Gaussian blur
    return cv2.GaussianBlur(image, (5, 5), 0)

def interpolate_image(image, scale_factor=4):
    height, width = image.shape[:2]
    new_dim = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('视频与图像增强')
        self.setGeometry(100, 100, 600, 400)
        
        self.input_path = ""
        self.is_video = False
        self.traditional_methods = []
        self.dl_model = 'RealBasicVSR_x4'
        
        layout = QGridLayout()

        self.radio_group = QButtonGroup(self)
        
        self.image_radio = QRadioButton("处理图片")
        self.image_radio.setChecked(True)
        self.image_radio.toggled.connect(self.set_input_type)
        self.radio_group.addButton(self.image_radio)
        layout.addWidget(self.image_radio, 0, 0)
        
        self.video_radio = QRadioButton("处理视频")
        self.video_radio.toggled.connect(self.set_input_type)
        self.radio_group.addButton(self.video_radio)
        layout.addWidget(self.video_radio, 0, 1)
        
        self.select_input_btn = QPushButton("选择输入文件夹/视频")
        self.select_input_btn.clicked.connect(self.select_input)
        layout.addWidget(self.select_input_btn, 1, 0, 1, 2)
        
        self.method_label = QLabel("使用传统处理方法:")
        layout.addWidget(self.method_label, 2, 0)
        
        self.traditional_cb = QCheckBox("启用")
        layout.addWidget(self.traditional_cb, 2, 1)
        
        self.dl_model_label = QLabel("选择深度学习模型:")
        layout.addWidget(self.dl_model_label, 3, 0)
        
        self.dl_model_combo = QComboBox()
        self.dl_model_combo.addItems(['RealBasicVSR_x4'])
        layout.addWidget(self.dl_model_combo, 3, 1, 1, 3)
        
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn, 4, 0, 1, 4)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar, 5, 0, 1, 4)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def set_input_type(self):
        self.is_video = self.video_radio.isChecked()

    def select_input(self):
        options = QFileDialog.Options()
        if self.is_video:
            self.input_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4);;所有文件 (*)", options=options)
        else:
            self.input_path = QFileDialog.getExistingDirectory(self, "选择输入文件夹", options=options)
        print(f"选择的输入路径: {self.input_path}")

    def start_processing(self):
        if not self.input_path:
            print("请输入路径")
            return
        
        self.traditional_methods = ['enhance_traditional'] if self.traditional_cb.isChecked() else []
        self.dl_model = self.dl_model_combo.currentText()
        
        self.worker = Worker(self.input_path, self.is_video, self.traditional_methods, self.dl_model)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def processing_finished(self):
        self.progress_bar.setValue(100)
        print("处理完成!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
