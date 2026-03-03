import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import datetime
import mediapipe as mp

# ================= MEDIAPIPE SEGMENTATION =================

mp_selfie = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=1)

def apply_background_blur(frame, intensity=0.8):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb)
    mask = results.segmentation_mask
    condition = mask > 0.5

    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    output = np.where(condition[..., None], frame, blurred)
    return output.astype(np.uint8)

# ================= FILTER FUNCTIONS =================

def apply_grayscale(img, intensity=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1.0 - intensity, gray, intensity, 0)

def apply_sepia(img, intensity=1.0):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.addWeighted(img, 1.0 - intensity, sepia, intensity, 0)

def apply_negative(img, intensity=1.0):
    neg = 255 - img
    return cv2.addWeighted(img, 1.0 - intensity, neg, intensity, 0)

def apply_blur(img, intensity=1.0):
    k = int(intensity * 35) // 2 * 2 + 1
    if k < 1:
        k = 1
    return cv2.GaussianBlur(img, (k, k), 0)

# ================= MAIN APP =================

class InstagramCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 Instagram Camera PRO Hybrid")
        self.root.geometry("1000x780")
        self.root.configure(bg="#121212")

        self.mode = "camera"
        self.loaded_image = None

        self.available_cams = self.detect_cameras()
        self.current_cam_index = self.available_cams[0]
        self.cap = cv2.VideoCapture(self.current_cam_index)

        self.filter_dict = {
            "Normal": None,
            "Grayscale": apply_grayscale,
            "Sepia": apply_sepia,
            "Negative": apply_negative,
            "Blur": apply_blur,
            "AI Background Blur": apply_background_blur
        }

        self.current_filter = "Normal"
        self.intensity = 0.7
        self.current_frame = None

        self.build_ui()
        self.update_frame()

    def detect_cameras(self, max_tested=5):
        cams = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                cams.append(i)
                cap.release()
            else:
                cap.release()
        return cams if cams else [0]

    def build_ui(self):
        self.video_label = tk.Label(self.root, bg="#121212")
        self.video_label.pack(pady=20)

        control_frame = tk.Frame(self.root, bg="#1f1f1f")
        control_frame.pack(pady=10, padx=20, fill="x")

        # ===== MODE BUTTONS =====
        tk.Button(control_frame, text="📷 Camera Mode",
                  command=self.switch_to_camera,
                  bg="#333", fg="white").grid(row=0, column=0, padx=5)

        tk.Button(control_frame, text="🖼 Load Image",
                  command=self.load_image,
                  bg="#333", fg="white").grid(row=0, column=1, padx=5)

        # ===== CAMERA SELECTOR =====
        tk.Label(control_frame, text="🎥 Kamera:",
                 bg="#1f1f1f", fg="white").grid(row=0, column=2)

        self.cam_selector = ttk.Combobox(control_frame,
                                         values=self.available_cams,
                                         state="readonly",
                                         width=5)
        self.cam_selector.set(self.current_cam_index)
        self.cam_selector.grid(row=0, column=3, padx=10)
        self.cam_selector.bind("<<ComboboxSelected>>", self.change_camera)

        # ===== FILTER SELECTOR =====
        tk.Label(control_frame, text="🎨 Filter:",
                 bg="#1f1f1f", fg="white").grid(row=1, column=0)

        self.filter_selector = ttk.Combobox(control_frame,
                                            values=list(self.filter_dict.keys()),
                                            state="readonly")
        self.filter_selector.set("Normal")
        self.filter_selector.grid(row=1, column=1, padx=10)
        self.filter_selector.bind("<<ComboboxSelected>>", self.change_filter)

        # ===== INTENSITY =====
        tk.Label(control_frame, text="🎚 Intensitas:",
                 bg="#1f1f1f", fg="white").grid(row=2, column=0)

        self.slider = tk.Scale(control_frame,
                               from_=0, to=1,
                               resolution=0.01,
                               orient="horizontal",
                               length=300,
                               bg="#1f1f1f",
                               fg="white",
                               highlightthickness=0,
                               troughcolor="#333333",
                               command=self.change_intensity)
        self.slider.set(self.intensity)
        self.slider.grid(row=2, column=1, pady=10)

        # ===== SAVE =====
        tk.Button(control_frame, text="📸 Save",
                  bg="#ff0055",
                  fg="white",
                  font=("Arial", 12, "bold"),
                  command=self.capture_image).grid(row=3, column=1, pady=10)

    # ================= MODE =================

    def switch_to_camera(self):
        self.mode = "camera"
        self.loaded_image = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.mode = "image"
            self.loaded_image = cv2.imread(path)

    def change_camera(self, event):
        new_index = int(self.cam_selector.get())
        self.cap.release()
        self.cap = cv2.VideoCapture(new_index)
        self.current_cam_index = new_index

    def change_filter(self, event):
        self.current_filter = self.filter_selector.get()

    def change_intensity(self, val):
        self.intensity = float(val)

    def capture_image(self):
        if self.current_frame is not None:
            filename = datetime.datetime.now().strftime("output_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, self.current_frame)
            print("✅ Disimpan:", filename)

    # ================= UPDATE LOOP =================

    def update_frame(self):
        if self.mode == "camera":
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(10, self.update_frame)
                return
            frame = cv2.flip(frame, 1)
        else:
            if self.loaded_image is None:
                self.root.after(10, self.update_frame)
                return
            frame = self.loaded_image.copy()

        filter_func = self.filter_dict[self.current_filter]
        if filter_func:
            frame = filter_func(frame, self.intensity)

        self.current_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = InstagramCameraApp(root)
    root.mainloop()