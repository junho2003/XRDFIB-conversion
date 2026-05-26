import os
import re
import glob
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

AXIS_SCALE = 120
VIEW_DIR = np.array([-0.34, 0.0, 0.94])
IMG_Y_DOWN_DIR = np.array([0.0, 1.0, 0.0])
KAPPA_TILT_DEG = 49.96768

def unit_vector(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector")
    return v / n

VIEW_DIR = unit_vector(VIEW_DIR)
IMG_Y_DOWN_DIR = unit_vector(IMG_Y_DOWN_DIR)
IMG_Y_DIR = -IMG_Y_DOWN_DIR
IMG_X_DIR = unit_vector(np.cross(IMG_Y_DIR, VIEW_DIR))

def x_rotation(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def y_rotation(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

def z_rotation(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def gonio_rotation_matrix(omega, theta, kappa, phi):
    om = math.radians(omega)
    ka = math.radians(kappa)
    ph = math.radians(phi)
    kt = math.radians(KAPPA_TILT_DEG)
    return z_rotation(-om) @ y_rotation(-kt) @ z_rotation(-ka) @ y_rotation(kt) @ z_rotation(-ph)

def read_jpr_angle_file(jpr_path):
    pattern = re.compile(
        r"#\s*(\d+)\s+"
        r"o:\s*([-+]?\d+(?:\.\d+)?)\s+"
        r"t:\s*([-+]?\d+(?:\.\d+)?)\s+"
        r"k:\s*([-+]?\d+(?:\.\d+)?)\s+"
        r"p:\s*([-+]?\d+(?:\.\d+)?)"
    )

    frames = []

    with open(jpr_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                frames.append({
                    "frame": int(m.group(1)),
                    "omega": float(m.group(2)),
                    "theta": float(m.group(3)),
                    "kappa": float(m.group(4)),
                    "phi": float(m.group(5))
                })

    if not frames:
        raise ValueError(f"No frame angle information found in:\n{jpr_path}")

    return frames

def find_images_for_jpr(movie_folder, movie_name):
    image_dict = {}
    image_files = []

    patterns = [
        os.path.join(movie_folder, f"{movie_name}*.jpg"),
        os.path.join(movie_folder, f"{movie_name}*.JPG"),
        os.path.join(movie_folder, f"{movie_name}*.jpeg"),
        os.path.join(movie_folder, f"{movie_name}*.JPEG")
    ]

    for p in patterns:
        image_files.extend(glob.glob(p))

    for path in image_files:
        name = os.path.basename(path)
        m = re.search(rf"^{re.escape(movie_name)}(\d+)\.(jpg|jpeg)$", name, re.IGNORECASE)

        if m:
            frame = int(m.group(1))
            image_dict[frame] = path

    if not image_dict:
        raise FileNotFoundError(
            f"No images found for movie:\n{movie_name}\n\n"
            f"Expected format:\n{movie_name}1.jpg, {movie_name}2.jpg, ..."
        )

    return image_dict

def find_latest_redlog_file(root_folder):

    log_folder = os.path.join(root_folder, "log")

    if not os.path.isdir(log_folder):
        raise FileNotFoundError(f"'log' folder not found in:\n{root_folder}")

    log_files = glob.glob(os.path.join(log_folder, "crysalispro_redLOG*.txt"))

    if not log_files:
        raise FileNotFoundError(f"No crysalispro_redLOG*.txt found in:\n{log_folder}")

    month_map = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
        "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
    }

    def extract_datetime(filepath):

        name = os.path.basename(filepath)

        m = re.search(
            r"redLOG\w{3}-(\w{3})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{4})",
            name
        )

        if not m:
            return (0,0,0,0,0)

        month_str = m.group(1)
        day = int(m.group(2))
        hour = int(m.group(3))
        minute = int(m.group(4))
        second = int(m.group(5))
        year = int(m.group(6))

        month = month_map.get(month_str, 0)

        return (year, month, day, hour, minute, second)

    return max(log_files, key=extract_datetime)

def read_ub_from_xrd_folder(root_folder):
    log_file = find_latest_redlog_file(root_folder)

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    no_constraint_idx = None

    for i in range(len(lines) - 1, -1, -1):
        if "No constraint" in lines[i]:
            no_constraint_idx = i
            break

    if no_constraint_idx is None:
        raise ValueError(f"'No constraint' block not found in:\n{log_file}")

    ub_idx = None

    for i in range(no_constraint_idx, min(no_constraint_idx + 20, len(lines))):
        if "UB - matrix:" in lines[i]:
            ub_idx = i
            break

    if ub_idx is None:
        raise ValueError(f"'UB - matrix:' not found after No constraint in:\n{log_file}")

    ub_rows = []

    for j in range(1, 4):
        row_line = lines[ub_idx + j].strip()
        main_part = row_line.split("(")[0].strip()
        nums = [float(x) for x in main_part.split()[:3]]
        ub_rows.append(nums)

    return np.array(ub_rows), log_file

def project_vector_to_image(v_lab):
    px = np.dot(v_lab, IMG_X_DIR)
    py = np.dot(v_lab, IMG_Y_DIR)
    return np.array([px, -py])

def draw_axis(draw, origin, vec2d, label, color):
    v = np.array(vec2d, dtype=float)
    length = np.linalg.norm(v)

    if length == 0:
        return

    end = origin + AXIS_SCALE * v
    draw.line([tuple(origin), tuple(end)], fill=color, width=4)

    v_unit = v / length
    angle = math.atan2(v_unit[1], v_unit[0])
    head_len = 15
    head_angle = math.radians(25)

    p1 = end - head_len * np.array([math.cos(angle - head_angle), math.sin(angle - head_angle)])
    p2 = end - head_len * np.array([math.cos(angle + head_angle), math.sin(angle + head_angle)])

    draw.polygon([tuple(end), tuple(p1), tuple(p2)], fill=color)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = None

    draw.text(tuple(end + 12 * v_unit), label, fill=color, font=font)

def draw_axes_on_image(img, UB_matrix, angles):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    origin = np.array([w / 2, h / 2])

    R = gonio_rotation_matrix(angles["omega"], angles["theta"], angles["kappa"], angles["phi"])

    a_lab = unit_vector(R @ (UB_matrix @ np.array([1.0, 0.0, 0.0])))
    b_lab = unit_vector(R @ (UB_matrix @ np.array([0.0, 1.0, 0.0])))
    c_lab = unit_vector(R @ (UB_matrix @ np.array([0.0, 0.0, 1.0])))

    a_2d = project_vector_to_image(a_lab)
    b_2d = project_vector_to_image(b_lab)
    c_2d = project_vector_to_image(c_lab)

    draw_axis(draw, origin, a_2d, "a*", "red")
    draw_axis(draw, origin, b_2d, "b*", "green")
    draw_axis(draw, origin, c_2d, "c*", "blue")

    info = (
        f"Frame {angles['frame']}   "
        f"o={angles['omega']:.2f}  "
        f"t={angles['theta']:.2f}  "
        f"k={angles['kappa']:.2f}  "
        f"p={angles['phi']:.2f}"
    )

    draw.rectangle([10, 10, 650, 42], fill=(0, 0, 0))
    draw.text((18, 18), info, fill="white")

    return img

class CrystalMovieViewer:

    def __init__(self, root):
        self.root = root
        self.root.title("Crystal Movie Viewer")

        self.xrd_folder = None
        self.movie_folder = None
        self.jpr_file = None
        self.movie_name = None
        self.image_dict = None
        self.frames = None
        self.UB_matrix = None
        self.log_file = None

        self.idx = 0
        self.tk_img = None
        self.frame_cache = {}
        self.max_cache_size = 30
        self.wheel_busy = False

        self.measure_points = []
        self.dragging_point = None
        self.measure_angle = None


        top_frame = tk.Frame(root)
        top_frame.pack(side="top", fill="x")

        tk.Button(top_frame, text="Open .jpr file", command=self.open_jpr_file).pack(side="left", padx=5, pady=5)
        tk.Button(top_frame, text="Previous", command=self.prev_frame).pack(side="left", padx=5, pady=5)
        tk.Button(top_frame, text="Next", command=self.next_frame).pack(side="left", padx=5, pady=5)

        self.status_label = tk.Label(top_frame, text="Load JPR movie", anchor="w")
        self.status_label.pack(side="left", padx=10)

        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(side="top", fill="both", expand=True)

        self.image_label.bind("<Button-1>", self.on_image_click)
        self.image_label.bind("<B1-Motion>", self.on_image_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_image_release)
        self.image_label.bind("<Button-3>", self.clear_measurement)

        self.root.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Button-4>", self.on_mousewheel_linux)
        self.root.bind("<Button-5>", self.on_mousewheel_linux)

    def open_jpr_file(self):
        jpr_file = filedialog.askopenfilename(
            title="Select JPR movie file",
            filetypes=[("JPR files", "*.jpr *.JPR"), ("All files", "*.*")]
        )

        if not jpr_file:
            return

        try:
            self.jpr_file = jpr_file
            self.movie_folder = os.path.dirname(jpr_file)
            self.movie_name = os.path.splitext(os.path.basename(jpr_file))[0]
            self.xrd_folder = os.path.dirname(self.movie_folder)

            self.UB_matrix, self.log_file = read_ub_from_xrd_folder(self.xrd_folder)
            self.frames = read_jpr_angle_file(jpr_file)
            self.image_dict = find_images_for_jpr(self.movie_folder, self.movie_name)

            self.idx = 0
            self.frame_cache = {}


            self.draw_current_frame()
            self.preload_nearby_frames()

            self.status_label.config(
                text=f"Movie: {self.movie_name} | Frames: {len(self.frames)} | UB: {os.path.basename(self.log_file)}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw_current_frame(self):
        if self.frames is None:
            messagebox.showwarning("Warning", "Please load JPR movie first.")
            return

        if self.UB_matrix is None:
            messagebox.showwarning("Warning", "UB matrix was not loaded.")
            return

        angles = self.frames[self.idx]
        frame_number = angles["frame"]

        if frame_number not in self.image_dict:
            messagebox.showerror("Error", f"No image found for frame {frame_number}")
            return

        cache_key = self.idx

        if cache_key in self.frame_cache:
            self.tk_img = self.frame_cache[cache_key]
        else:
            img_path = self.image_dict[frame_number]
            img = Image.open(img_path)
            img = draw_axes_on_image(img, self.UB_matrix, angles)

            max_w = 1100
            max_h = 800
            img.thumbnail((max_w, max_h))

            img = self.draw_measurement_overlay(img)

            self.tk_img = ImageTk.PhotoImage(img)
            self.frame_cache[cache_key] = self.tk_img

            if len(self.frame_cache) > self.max_cache_size:
                oldest_key = sorted(self.frame_cache.keys())[0]
                del self.frame_cache[oldest_key]

        self.image_label.config(image=self.tk_img)

        img_path = self.image_dict[frame_number]
        self.status_label.config(
            text=f"{self.idx + 1}/{len(self.frames)} | {os.path.basename(img_path)} | "
                 f"o={angles['omega']:.2f}, t={angles['theta']:.2f}, "
                 f"k={angles['kappa']:.2f}, p={angles['phi']:.2f}"
        )
        
        self.preload_nearby_frames()

    def next_frame(self):
        if self.frames is None:
            return
    
        self.measure_points = []
        self.measure_angle = None   
    
        self.idx = (self.idx + 1) % len(self.frames)
    
        self.draw_current_frame()

    def prev_frame(self):
        if self.frames is None:
            return
    
        self.measure_points = []
        self.measure_angle = None
    
        self.idx = (self.idx - 1) % len(self.frames)
    
        self.draw_current_frame()

    def on_image_click(self, event):
        if self.frames is None:
            return
    
        x, y = event.x, event.y
    
        # 기존 점 근처 클릭하면 drag 시작
        for i, (px, py) in enumerate(self.measure_points):
            if (x - px)**2 + (y - py)**2 < 12**2:
                self.dragging_point = i
                return
    
        # 점이 2개 미만이면 새 점 추가
        if len(self.measure_points) < 2:
            self.measure_points.append((x, y))
        else:
            # 이미 선분이 있으면 새 첫 점으로 다시 시작
            self.measure_points = [(x, y)]
    
        self.frame_cache.pop(self.idx, None)
        self.draw_current_frame()
    
    def on_image_drag(self, event):
        if self.dragging_point is None:
            return
    
        self.measure_points[self.dragging_point] = (event.x, event.y)
    
        self.frame_cache.pop(self.idx, None)
        self.draw_current_frame()


    def on_image_release(self, event):
        self.dragging_point = None
    
    def clear_measurement(self, event=None):
        self.measure_points = []
        self.dragging_point = None
    
        self.frame_cache.pop(self.idx, None)
        self.draw_current_frame()

    def draw_measurement_overlay(self, img):
        if not self.measure_points:
            return img
    
        draw = ImageDraw.Draw(img)
    
        for x, y in self.measure_points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="yellow")
    
        if len(self.measure_points) == 2:
            x1, y1 = self.measure_points[0]
            x2, y2 = self.measure_points[1]
    
            draw.line([x1, y1, x2, y2], fill="yellow", width=1)
    
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(-dy, dx))
    
            text = f"Angle : {angle:.2f} deg"

            w, h = img.size
            
            draw.rectangle([w - 260, 10, w - 10, 50], fill="black")
            draw.text((w - 245, 18), text, fill="yellow")
    
        return img


    def preload_nearby_frames(self):
        if self.frames is None or self.UB_matrix is None:
            return
    
        n = len(self.frames)
        preload_range = 10
    
        for offset in range(-preload_range, preload_range + 1):
    
            i = (self.idx + offset) % n
    
            if i in self.frame_cache:
                continue
    
            angles = self.frames[i]
            frame_number = angles["frame"]
    
            if frame_number not in self.image_dict:
                continue
    
            try:
                img_path = self.image_dict[frame_number]
    
                img = Image.open(img_path)
                img = draw_axes_on_image(img, self.UB_matrix, angles)
    
                img.thumbnail((1100, 800))
                
                img = self.draw_measurement_overlay(img)
                
                self.frame_cache[i] = ImageTk.PhotoImage(img)
    
            except:
                pass
    
        keep = set((self.idx + offset) % n for offset in range(-20, 21))
    
        for key in list(self.frame_cache.keys()):
            if key not in keep:
                del self.frame_cache[key]

    def on_mousewheel(self, event):
        if self.wheel_busy:
            return

        self.wheel_busy = True

        if event.delta < 0:
            self.next_frame()
        else:
            self.prev_frame()

        self.root.after(30, self.release_wheel)

    def release_wheel(self):
        self.wheel_busy = False

    def on_mousewheel_linux(self, event):
        if event.num == 5:
            self.next_frame()
        elif event.num == 4:
            self.prev_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = CrystalMovieViewer(root)
    root.mainloop()