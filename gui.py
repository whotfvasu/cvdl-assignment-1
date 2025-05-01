import sys
import cv2
import torch
import time
import torchvision.transforms as T
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QMessageBox
)
import torch.hub
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QVBoxLayout,
#     QWidget, QHBoxLayout, QMessageBox
# )
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

def convert_cv_qt(cv_img):
    """Convert OpenCV image (BGR) to QImage (RGB)."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

def get_device():
    """Return the best available device: CUDA or CPU for Windows."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class LiveFeedWorker(QThread):
    changePixmap = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self._running = True
        self._paused = False  # For pause/resume functionality

    def toggle_pause(self):
        self._paused = not self._paused

    def run(self):
        if self.model_type == "YOLOv8":
            self.run_yolo()
        elif self.model_type == "Faster R-CNN":
            self.run_faster_rcnn()
        elif self.model_type == "YOLOv5":
            self.run_yolov5()
        self.finished.emit()


    def run_yolo(self):
        device = get_device()
        model = YOLO('yolov8n.pt')
        model.to(device)
        CLASS_NAMES = model.names

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        prev_time = time.time()
        while self._running:
            if self._paused:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = CLASS_NAMES.get(class_id, "Unknown")
                    label = f"{class_name} | {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
            q_img = convert_cv_qt(frame)
            self.changePixmap.emit(q_img)
            self.msleep(30)

        cap.release()

    def run_yolov5(self):
        device = get_device()
        try:
            # Try to load YOLOv5 from torch.hub (online)
            print("Loading YOLOv5 model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            print("YOLOv5 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv5 from torch.hub: {e}")
            # Fall back to YOLOv8 if YOLOv5 fails to load
            print("Falling back to YOLOv8")
            self.run_yolo()
            return
            
        model.to(device)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        prev_time = time.time()
        while self._running:
            if self._paused:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # YOLOv5 inference
            results = model(frame)
            
            # Get the annotated frame but make a copy of it first
            results_frame = results.render()[0].copy()  # Make a copy to make it writable
            
            cv2.putText(results_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            q_img = convert_cv_qt(results_frame)
            self.changePixmap.emit(q_img)
            self.msleep(30)

        cap.release()

        def run_faster_rcnn(self):
            device = get_device()
            rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
            rcnn_model.to(device)
            rcnn_model.eval()

            COCO_LABELS = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'shoe',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

            def preprocess_image(image):
                transform = T.Compose([T.ToTensor()])
                return transform(image).unsqueeze(0).to(device)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            while self._running:
                if self._paused:
                    self.msleep(100)
                    continue

                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                image_tensor = preprocess_image(frame)
                with torch.no_grad():
                    outputs = rcnn_model(image_tensor)

                for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                    if score > 0.5:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        class_id = int(label.item())
                        class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
                        confidence = round(float(score.item()), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 0), 2)
                        cv2.putText(frame, f"{class_name} | {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)

                total_time = time.time() - start_time
                fps = 1.0 / total_time if total_time > 0 else 0.0

                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                q_img = convert_cv_qt(frame)
                self.changePixmap.emit(q_img)
                self.msleep(30)

            cap.release()

        def stop(self):
            self._running = False

class CompareWorker(QThread):
    changePixmapYOLO = pyqtSignal(QImage)
    changePixmapRCNN = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        device = get_device()
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.to(device)
        yolo_class_names = yolo_model.names

        rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        rcnn_model.to(device)
        rcnn_model.eval()
        COCO_LABELS = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'shoe',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
            'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        def preprocess_image(image):
            transform = T.Compose([T.ToTensor()])
            return transform(image).unsqueeze(0).to(device)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame_yolo = frame.copy()
            frame_rcnn = frame.copy()

            # Process YOLO
            results = yolo_model(frame_yolo)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = yolo_class_names.get(class_id, "Unknown")
                    label = f"{class_name} | {conf:.2f}"
                    cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame_yolo, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            cv2.putText(frame_yolo, "YOLOv8", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
            q_img_yolo = convert_cv_qt(frame_yolo)
            self.changePixmapYOLO.emit(q_img_yolo)

            # Process Faster R-CNN
            image_tensor = preprocess_image(frame_rcnn)
            with torch.no_grad():
                outputs = rcnn_model(image_tensor)
            for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                if score > 0.5:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_id = int(label.item())
                    class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
                    confidence = round(float(score.item()), 2)
                    cv2.rectangle(frame_rcnn, (x1, y1), (x2, y2), (200, 0, 0), 2)
                    cv2.putText(frame_rcnn, f"{class_name} | {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
            cv2.putText(frame_rcnn, "Faster R-CNN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            q_img_rcnn = convert_cv_qt(frame_rcnn)
            self.changePixmapRCNN.emit(q_img_rcnn)

            self.msleep(30)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False

class LiveFeedWindow(QMainWindow):
    def __init__(self, model_type, main_window):
        super().__init__()
        self.setWindowTitle(f"Live Feed - {model_type}")
        self.setGeometry(150, 150, 800, 600)
        self.main_window = main_window
        self.current_frame = None  # For snapshot functionality

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Additional Buttons: Pause and Snapshot
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.pause_button.clicked.connect(self.toggle_pause)

        self.snapshot_button = QPushButton("Snapshot", self)
        self.snapshot_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.snapshot_button.clicked.connect(self.take_snapshot)

        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.back_button.clicked.connect(self.back)

        # Layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.snapshot_button)
        button_layout.addWidget(self.back_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = LiveFeedWorker(model_type)
        self.worker.changePixmap.connect(self.setImage)
        self.worker.start()

    def setImage(self, image):
        self.current_frame = image
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def toggle_pause(self):
        self.worker.toggle_pause()
        if self.pause_button.text() == "Pause":
            self.pause_button.setText("Resume")
        else:
            self.pause_button.setText("Pause")

    def take_snapshot(self):
        if self.current_frame:
            pixmap = QPixmap.fromImage(self.current_frame)
            timestamp = int(time.time())
            filename = f"snapshot_{timestamp}.png"
            if pixmap.save(filename):
                QMessageBox.information(self, "Snapshot", f"Snapshot saved as {filename}")
            else:
                QMessageBox.warning(self, "Snapshot", "Failed to save snapshot.")
        else:
            QMessageBox.warning(self, "Snapshot", "No frame available to capture.")

    def back(self):
        self.worker.stop()
        self.worker.wait()
        self.close()
        self.main_window.show()

class CompareWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Compare: YOLOv8 vs Faster R-CNN")
        self.setGeometry(150, 150, 1200, 600)
        self.main_window = main_window

        self.label_yolo = QLabel(self)
        self.label_yolo.setAlignment(Qt.AlignCenter)
        self.label_rcnn = QLabel(self)
        self.label_rcnn.setAlignment(Qt.AlignCenter)

        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.back_button.clicked.connect(self.back)

        feeds_layout = QHBoxLayout()
        feeds_layout.addWidget(self.label_yolo)
        feeds_layout.addWidget(self.label_rcnn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(feeds_layout)
        main_layout.addWidget(self.back_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.worker = CompareWorker()
        self.worker.changePixmapYOLO.connect(self.setImageYOLO)
        self.worker.changePixmapRCNN.connect(self.setImageRCNN)
        self.worker.start()

    def setImageYOLO(self, image):
        self.label_yolo.setPixmap(QPixmap.fromImage(image))

    def setImageRCNN(self, image):
        self.label_rcnn.setPixmap(QPixmap.fromImage(image))

    def back(self):
        self.worker.stop()
        self.worker.wait()
        self.close()
        self.main_window.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real Time Object Detection")
        self.setGeometry(100, 100, 550, 350)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2f;
            }
            QLabel#titleLabel {
                font-size: 36px;
                font-weight: bold;
                color: #f0f0f0;
            }
            QComboBox, QPushButton {
                font-size: 20px;
                padding: 10px;
            }
            QComboBox {
                background-color: #2e2e42;
                color: #ffffff;
                border: 1px solid #3a3a55;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #3a86ff;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #265df2;
            }
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        self.title_label = QLabel("Real Time Object Detection", self)
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        self.combo = QComboBox()
        self.combo.addItems([
            "Real-Time Detection (YOLOv8)",
            "Real-Time Detection (YOLOv5)",
            "Real-Time Detection (Faster R-CNN)",
            "Compare Models"
        ])
        layout.addWidget(self.combo)

        self.button = QPushButton("Start")
        self.button.clicked.connect(self.start_model)
        layout.addWidget(self.button)

        self.setCentralWidget(container)

    def start_model(self):
        model_choice = self.combo.currentText()
        self.hide()
        if "Compare" in model_choice:
            self.compare_window = CompareWindow(self)
            self.compare_window.show()
        else:
            if "YOLOv8" in model_choice:
                model_type = "YOLOv8"
            elif "YOLOv5" in model_choice:
                model_type = "YOLOv5"
            elif "Faster" in model_choice:
                model_type = "Faster R-CNN"
            else:
                model_type = model_choice

            self.live_feed_window = LiveFeedWindow(model_type, self)
            self.live_feed_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
