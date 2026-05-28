import math
import numpy as np
import os
import glob
import subprocess
import sys
import re
from PIL import Image, ImageDraw, ImageFont


class Operations():
    """SEM is parallel to the Z axis"""
    """FIB is 52 degrees tilted from Z axis towards -Y axis"""
    """SEM window right side is towards -X axis"""

    def __init__(self):
        self.UB_matrix = None
        self.log_file = None
        self.unit_cell_text = None
        self.result_text = None
        self.axis_scale = 180
        self.view_dir = np.array([-0.34, 0.0, 0.94])
        self.img_y_down_dir = np.array([0.0, 1.0, 0.0])
        self.kappa_tilt_deg = 49.96768

        # Setup class properties initialized via the unit_vector method
        self.view_dir_unit = self.unit_vector(self.view_dir)
        self.img_y_unit = self.unit_vector(self.img_y_down_dir)
        self.img_y_dir = -self.img_y_unit
        self.img_x_dir = self.unit_vector(np.cross(self.img_y_dir, self.view_dir_unit))
        self.frames = None
   
    # Pre-calculate derived vectors safely during class definition
    def unit_vector(self, v):
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError("Zero vector")
        return v / n

    def get_Rx(self, theta):
        return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

    def get_Ry(self, theta):
        return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])

    def get_Rz(self, theta):
        return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

    def gonio_rotation_matrix(self, omega, theta, kappa, phi):
        om = math.radians(omega)
        ka = math.radians(kappa)
        ph = math.radians(phi)
        kt = math.radians(self.kappa_tilt_deg)
        return self.get_Rz(-om) @ self.get_Ry(-kt) @ self.get_Rz(-ka) @ self.get_Ry(kt) @ self.get_Rz(-ph)

    def project_vector_to_image(self, v_lab):
        px = np.dot(v_lab, self.img_x_dir)
        py = np.dot(v_lab, self.img_y_dir)
        return np.array([px, -py])

    def draw_axis(self, draw, origin, vec2d, label, color):
        v = np.array(vec2d, dtype=float)
        length = np.linalg.norm(v)
        if length == 0:
            return
        end = origin + self.axis_scale * v
        draw.line([tuple(origin), tuple(end)], fill=color, width=4)
        v_unit = v / length
        angle = math.atan2(v_unit[1], v_unit[0])
        head_len = 22
        head_angle = math.radians(25)
        p1 = end - head_len * np.array([math.cos(angle - head_angle), math.sin(angle - head_angle)])
        p2 = end - head_len * np.array([math.cos(angle + head_angle), math.sin(angle + head_angle)])
        draw.polygon([tuple(end), tuple(p1), tuple(p2)], fill=color)
        #font = ImageFont.truetype("arial.ttf", 100)
        font = ImageFont.load_default(size=50)
        draw.text(tuple(end + 18 * v_unit), label, fill=color, font=font)

    def draw_axes_on_image(self, img, UB_matrix, angles):
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        origin = np.array([w / 2, h / 2])
        R = self.gonio_rotation_matrix(angles["omega"], angles["theta"], angles["kappa"], angles["phi"])
        a_lab = self.unit_vector(R @ (UB_matrix @ np.array([1.0, 0.0, 0.0])))
        b_lab = self.unit_vector(R @ (UB_matrix @ np.array([0.0, 1.0, 0.0])))
        c_lab = self.unit_vector(R @ (UB_matrix @ np.array([0.0, 0.0, 1.0])))
        a_2d = self.project_vector_to_image(a_lab)
        b_2d = self.project_vector_to_image(b_lab)
        c_2d = self.project_vector_to_image(c_lab)
        self.draw_axis(draw, origin, a_2d, "a*", "red")
        self.draw_axis(draw, origin, b_2d, "b*", "green")
        self.draw_axis(draw, origin, c_2d, "c*", "blue")
        return img

    def read_jpr_angle_file(self, jpr_path):
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
        self.frames = frames

    @staticmethod
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

# --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------

    def script_path(self, filename):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            filename
        )

    def unit_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Zero vector is not allowed.")
        return vector / norm

    def proj_z(self, vector, z):
        z_norm = np.sqrt(np.sum(z ** 2))
        return (np.dot(vector, z) / z_norm ** 2) * z

    def x_rotation(self, vector, theta):
        R = self.get_Rx(theta)
        return np.dot(R, vector)

    def y_rotation(self, vector, theta):
        R = self.get_Ry(theta)
        return np.dot(R, vector)

    def z_rotation(self, vector, theta):
        R = self.get_Rz(theta)
        return np.dot(R, vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        c = np.array([0, 0, 1])
        theta = np.degrees(
            np.arccos(
                np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            )
        )
        if np.dot(np.cross(v1_u, v2_u), c) < 0:
            return round(theta, 2)
        else:
            return round(360 - theta, 2)

    def tiltstage(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        phi = np.degrees(
            np.arccos(
                np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            )
        )
        return round(phi, 2)

    # --------------------------------------------------------
    # File handling
    # --------------------------------------------------------

    def find_latest_redlog_file(self, root_folder):
        log_folder = os.path.join(root_folder, "log")
        if not os.path.isdir(log_folder):
            raise FileNotFoundError(
                f"'log' folder not found in:\n{root_folder}"
            )
        log_files = glob.glob(
            os.path.join(log_folder, "crysalispro_redLOG*.txt")
        )
        if not log_files:
            raise FileNotFoundError(
                f"No crysalispro_redLOG*.txt found in:\n{log_folder}"
            )
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3,
            "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9,
            "Oct": 10, "Nov": 11, "Dec": 12
        }
        def extract_datetime(filepath):
            name = os.path.basename(filepath)
            m = re.search(
                r"redLOG\w{3}-(\w{3})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{4})",
                name
            )
            if not m:
                return (0, 0, 0, 0, 0)
            month_str = m.group(1)
            day = int(m.group(2))
            hour = int(m.group(3))
            minute = int(m.group(4))
            second = int(m.group(5))
            year = int(m.group(6))
            month = month_map.get(month_str, 0)
            return (
                year,
                month,
                day,
                hour,
                minute,
                second
            )
        self.log_file = max(log_files, key=extract_datetime)

    def extract_unit_cell_after_constraint(self, lines, start_idx):
        for i in range(start_idx, len(lines)):
            if "UB fit with" in lines[i]:
                for j in range(i + 1, min(i + 8, len(lines))):
                    if "unit cell:" in lines[j]:
                        if j + 3 >= len(lines):
                            continue
                        line1 = lines[j + 1].strip()
                        line2 = lines[j + 2].strip()
                        line3 = lines[j + 3].strip()
                        if "V =" not in line3:
                            continue
                        self.unit_cell_text = (
                            #f"UB fit:\n"
                            #f"{lines[i].strip()}\n\n"
                            f"unit cell:\n"
                            f"{line1}\n"
                            f"{line2}\n"
                            f"{line3}"
                        )
                        return

        self.unit_cell_text = "No unit cell information found."

    def read_ub_from_xrd_folder(self, xrd_folder):
        self.find_latest_redlog_file(xrd_folder)
        with open(
            self.log_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as fin:

            lines = fin.readlines()

        no_constraint_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if "No constraint" in lines[i]:
                no_constraint_idx = i
                break

        if no_constraint_idx is None:
            raise ValueError(
                f"'No constraint' block not found in:\n{self.log_file}"
            )

        ub_idx = None
        for i in range(no_constraint_idx, len(lines)):
            if "UB - matrix:" in lines[i]:
                ub_idx = i
                break
        if ub_idx is None:
            raise ValueError(
                f"'UB - matrix' not found after No constraint"
            )

        ub_rows = []

        for j in range(1, 4):
            row_line = lines[ub_idx + j].strip()
            main_part = row_line.split("(")[0].strip()
            nums = [
                float(x)
                for x in main_part.split()[:3]
            ]
            ub_rows.append(nums)

        self.UB_matrix = np.array(ub_rows)
        self.extract_unit_cell_after_constraint(lines, ub_idx)

    def calculate_all(self,u,lamella,angle,reference,ub_file):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        
        if isinstance(reference, float):
            ref_output = f"Rotate stage relatively by: {round(reference,1)}"
        else:
            reference_xyz = self.UB_matrix @ reference
            ref_output = f"Rotate stage relatively by: {round(self.angle_between(-x, reference_xyz - self.proj_z(reference_xyz, z)),1)}"
        u_xyz = self.unit_vector(self.UB_matrix @ u)
        lamella_xyz = self.unit_vector(self.UB_matrix @ lamella)
        p_xyz = self.unit_vector(np.cross(u_xyz, lamella_xyz))

        alpha1 = self.angle_between(u_xyz - self.proj_z(u_xyz, z),-y)
        beta1 = self.tiltstage(u_xyz, z)

        P0a_new = self.unit_vector(self.x_rotation(self.z_rotation(lamella_xyz,math.radians(360 - alpha1)),math.radians(360 - beta1)))
        P0_BR = round(self.angle_between(-x, P0a_new),1)

        p1_xyz = self.unit_vector(math.tan(math.radians(angle)) * p_xyz + u_xyz)
        p2_xyz = self.unit_vector(math.tan(math.radians(-angle)) * p_xyz + u_xyz)
        p3_xyz = self.unit_vector(math.tan(math.radians(-60)) * p_xyz + u_xyz)

        alpha2_A = self.angle_between(p1_xyz - self.proj_z(p1_xyz, z), -y)
        beta2_A = self.tiltstage(p1_xyz, z)
        perp2_A = self.tiltstage(y, self.unit_vector(self.x_rotation(self.z_rotation(p_xyz, math.radians(360 - alpha2_A)), math.radians(360 - beta2_A))))
        fibtilt2_A = round(52 - beta2_A, 1)
        P1a_new_A = self.unit_vector(self.x_rotation(self.z_rotation(-lamella_xyz, math.radians(360 - alpha2_A)), math.radians(beta2_A)))
        P1_BR_A = round(self.angle_between(-x, P1a_new_A), 1)

        alpha2_B = self.angle_between(p1_xyz - self.proj_z(p1_xyz, z), +y)
        beta2_B = self.tiltstage(p1_xyz, z)
        perp2_B = self.tiltstage(y, self.unit_vector(self.x_rotation(self.z_rotation(p_xyz, math.radians(360 - alpha2_B)), math.radians(360 + beta2_B))))
        fibtilt2_B = round(52 + beta2_B, 1)
        P1a_new_B = self.unit_vector(self.x_rotation(self.z_rotation(-lamella_xyz, math.radians(360 - alpha2_B)), math.radians(-beta2_B)))
        P1_BR_B = round(self.angle_between(-x, P1a_new_B), 1)

        alpha3_A = self.angle_between(p2_xyz - self.proj_z(p2_xyz, z), -y)
        beta3_A = self.tiltstage(p2_xyz, z)
        perp3_A = self.tiltstage(y, self.unit_vector(self.x_rotation(self.z_rotation(-p_xyz, math.radians(360 - alpha3_A)), math.radians(360 - beta3_A))))
        fibtilt3_A = round(52 - beta3_A, 1)
        P2a_new_A = self.unit_vector(self.x_rotation(self.z_rotation(lamella_xyz, math.radians(360 - alpha3_A)), math.radians(beta3_A)))
        P2_BR_A = round(self.angle_between(-x, P2a_new_A), 1)

        alpha3_B = self.angle_between(p2_xyz - self.proj_z(p2_xyz, z), +y)
        beta3_B = self.tiltstage(p2_xyz, z)
        perp3_B = self.tiltstage(y, self.unit_vector(self.x_rotation(self.z_rotation(-p_xyz, math.radians(360 - alpha3_B)), math.radians(360 + beta3_B))))
        fibtilt3_B = round(52 + beta3_B, 1)
        P2a_new_B = self.unit_vector(self.x_rotation(self.z_rotation(lamella_xyz, math.radians(360 - alpha3_B)), math.radians(-beta3_B)))
        P2_BR_B = round(self.angle_between(-x, P2a_new_B), 1)


        def two_col(title, A, B):
            return f"""
    {title}
    {'Config A':<42}{'Config B':<42}
    {'-'*40}  {'-'*40}
    {f"{'Stage rotation':<20}: {A['alpha']}":<42}{f"{'Stage rotation':<20}: {B['alpha']}":<42}
    {f"{'Stage tilt':<20}: {A['fibtilt']}":<42}{f"{'Stage tilt':<20}: {B['fibtilt']}":<42}
    {f"{'Pattern rotation':<20}: {A['beam']}":<42}{f"{'Pattern rotation':<20}: {B['beam']}":<42}
    {f"{'Face towards':<20}: {A['perp']}":<42}{f"{'Face towards':<20}: {B['perp']}":<42}
    """

        side1_A = {"alpha": round(alpha2_A,1), "fibtilt": fibtilt2_A, "beam": P1_BR_A, "perp": round(perp2_A,1)}
        side1_B = {"alpha": round(alpha2_B,1), "fibtilt": fibtilt2_B, "beam": P1_BR_B, "perp": round(perp2_B,1)}
        side2_A = {"alpha": round(alpha3_A,1), "fibtilt": fibtilt3_A, "beam": P2_BR_A, "perp": round(perp3_A,1)}
        side2_B = {"alpha": round(alpha3_B,1), "fibtilt": fibtilt3_B, "beam": P2_BR_B, "perp": round(perp3_B,1)}



        self.result_text = f"""
    ========== Follow these instructions ==========

    1. Align your reference vector towards right horizontal direction of SEM 
    """
        self.result_text += '2. ' + ref_output
        self.result_text += f"""
    3. Save current position as starting position

    ====================

    !!!All patterning angles below must be set from the saved starting position!!!

    --------------------------------

    Trench cuts:

    Relative stage rotation : {round(alpha1,1)}
    Absolute stage tilt     : {round(52 - beta1, 1)}
    Pattern rotation        : {P0_BR}
    
    --------------------------------
    
    Polishing:

    """
        self.result_text += two_col("5. Polishing side1:", side1_A, side1_B)
        self.result_text += two_col("6. Polishing side2:", side2_A, side2_B)

        self.result_text += f"""

    --------------------------------
    Log file used:
    {self.log_file}
    """

