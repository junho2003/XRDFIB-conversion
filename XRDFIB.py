import math
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import glob
import subprocess
import sys
import re

"""SEM is parallel to the Z axis"""
"""FIB is 52degrees tilted from Z axis towards -Y axis"""
"""SEM window right side is towards -X axis"""

def script_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero vector is not allowed.")
    return vector / norm

def proj_z(vector, z):
    z_norm = np.sqrt(np.sum(z**2))
    return (np.dot(vector, z) / z_norm**2) * z

"""x,y,z rotation is counter-clockwise seen from positive x,y,z axes"""
def x_rotation(vector, theta):
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.dot(R, vector)

def y_rotation(vector, theta):
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)

def z_rotation(vector, theta):
    R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    return np.dot(R, vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' in clockwise direction::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    c = np.array([0,0,1])

    theta = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    if np.dot(np.cross(v1_u, v2_u), c) < 0:
        return round(theta, 2)
    else:
        return round(360 - theta, 2)

def tiltstage(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    phi = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    return round(phi, 2)

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


def read_ub_from_xrd_folder(xrd_folder):
    log_file = find_latest_redlog_file(xrd_folder)

    with open(log_file, "r", encoding="utf-8", errors="ignore") as fin:
        lines = fin.readlines()

    no_constraint_idx = None
    
    for i in range(len(lines) - 1, -1, -1):
    
        if "No constraint" in lines[i]:
            no_constraint_idx = i
            break
    
    if no_constraint_idx is None:
        raise ValueError(f"'No constraint' block not found in:\n{log_file}")

    ub_idx = None

    for i in range(no_constraint_idx, len(lines)):
        if "UB - matrix:" in lines[i]:
            ub_idx = i
            break

    if ub_idx is None:
        raise ValueError(f"'UB - matrix' not found after No constraint in:\n{log_file}")

    ub_rows = []

    for j in range(1, 4):
        row_line = lines[ub_idx + j].strip()
        main_part = row_line.split("(")[0].strip()
        nums = [float(x) for x in main_part.split()[:3]]
        ub_rows.append(nums)

    UB_matrix = np.array(ub_rows)

    unit_cell_text = extract_unit_cell_after_constraint(lines, ub_idx)

    ub_text = (
    f"UB matrix:\n"
    f"{UB_matrix[0,0]: .6f} {UB_matrix[0,1]: .6f} {UB_matrix[0,2]: .6f}\n"
    f"{UB_matrix[1,0]: .6f} {UB_matrix[1,1]: .6f} {UB_matrix[1,2]: .6f}\n"
    f"{UB_matrix[2,0]: .6f} {UB_matrix[2,1]: .6f} {UB_matrix[2,2]: .6f}"
    )

    full_info = ub_text + "\n\n" + unit_cell_text

    return UB_matrix, log_file, full_info


def extract_unit_cell_after_constraint(lines, start_idx):

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

                    return (
                        f"UB fit:\n"
                        f"{lines[i].strip()}\n\n"
                        f"unit cell:\n"
                        f"{line1}\n"
                        f"{line2}\n"
                        f"{line3}"
                    )

    return "No unit cell information found."

def calculate_all(u, lamella, angle, reference, ub_file):

    UB_matrix, log_file, unit_cell_text = read_ub_from_xrd_folder(ub_file)

    x = np.array([1.0,0.0,0.0])
    y = np.array([0.0,1.0,0.0])
    z = np.array([0.0,0.0,1.0])

    u_xyz = unit_vector(UB_matrix @ u)
    lamella_xyz = unit_vector(UB_matrix @ lamella)
    p_xyz = unit_vector(np.cross(u_xyz, lamella_xyz))

    p1_xyz = unit_vector(math.tan(math.radians(angle)) * p_xyz + u_xyz)
    p2_xyz = unit_vector(math.tan(math.radians(-angle)) * p_xyz + u_xyz)
    p3_xyz = unit_vector(math.tan(math.radians(-60)) * p_xyz + u_xyz)

    reference_xyz = UB_matrix @ reference

    alpha1 = angle_between(u_xyz - proj_z(u_xyz, z), -y)
    beta1 = tiltstage(u_xyz, z)

    P0a_new = unit_vector(x_rotation(z_rotation(lamella_xyz, math.radians(360 - alpha1)), math.radians(360 - beta1)))
    P0_BR = round(angle_between(-x, P0a_new), 1)

    alpha2_A = angle_between(p1_xyz - proj_z(p1_xyz, z), -y)
    beta2_A = tiltstage(p1_xyz, z)
    perp2_A = tiltstage(y, unit_vector(x_rotation(z_rotation(p_xyz, math.radians(360 - alpha2_A)), math.radians(360 - beta2_A))))
    fibtilt2_A = round(52 - beta2_A, 1)
    P1a_new_A = unit_vector(x_rotation(z_rotation(-lamella_xyz, math.radians(360 - alpha2_A)), math.radians(beta2_A)))
    P1_BR_A = round(angle_between(-x, P1a_new_A), 1)

    alpha2_B = angle_between(p1_xyz - proj_z(p1_xyz, z), +y)
    beta2_B = tiltstage(p1_xyz, z)
    perp2_B = tiltstage(y, unit_vector(x_rotation(z_rotation(p_xyz, math.radians(360 - alpha2_B)), math.radians(360 + beta2_B))))
    fibtilt2_B = round(52 + beta2_B, 1)
    P1a_new_B = unit_vector(x_rotation(z_rotation(-lamella_xyz, math.radians(360 - alpha2_B)), math.radians(-beta2_B)))
    P1_BR_B = round(angle_between(-x, P1a_new_B), 1)

    alpha3_A = angle_between(p2_xyz - proj_z(p2_xyz, z), -y)
    beta3_A = tiltstage(p2_xyz, z)
    perp3_A = tiltstage(y, unit_vector(x_rotation(z_rotation(-p_xyz, math.radians(360 - alpha3_A)), math.radians(360 - beta3_A))))
    fibtilt3_A = round(52 - beta3_A, 1)
    P2a_new_A = unit_vector(x_rotation(z_rotation(lamella_xyz, math.radians(360 - alpha3_A)), math.radians(beta3_A)))
    P2_BR_A = round(angle_between(-x, P2a_new_A), 1)

    alpha3_B = angle_between(p2_xyz - proj_z(p2_xyz, z), +y)
    beta3_B = tiltstage(p2_xyz, z)
    perp3_B = tiltstage(y, unit_vector(x_rotation(z_rotation(-p_xyz, math.radians(360 - alpha3_B)), math.radians(360 + beta3_B))))
    fibtilt3_B = round(52 + beta3_B, 1)
    P2a_new_B = unit_vector(x_rotation(z_rotation(lamella_xyz, math.radians(360 - alpha3_B)), math.radians(-beta3_B)))
    P2_BR_B = round(angle_between(-x, P2a_new_B), 1)

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

    result_text = f"""
========== Follow instruction ==========
1. Align your reference vector towards right horizontal direction of SEM
2. Rotate stage relatively by: {round(angle_between(-x, reference_xyz - proj_z(reference_xyz, z)),1)}
3. Save current position as starting position

!!!Below patterning angles must be set from the saved starting position!!!

--------------------------------
4. Trench cut
{'Stage rotation':<20}: {round(alpha1,1)}
{'Stage tilt':<20}: {round(52 - beta1, 1)}
{'Pattern rotation':<20}: {P0_BR}
"""

    result_text += two_col("5. Polishing side1", side1_A, side1_B)
    result_text += two_col("6. Polishing side2", side2_A, side2_B)

    result_text += f"""

--------------------------------
Log file used:
{log_file}
"""

    return result_text, unit_cell_text

class App:

    def __init__(self, root):

        self.root = root
        self.root.title("FIB / XRD Angle Calculator")

        self.ub_file = tk.StringVar()
        
        tk.Label(root, text="XRD data(exp) folder").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(root, textvariable=self.ub_file, width=45).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        self.make_row("reference (h,k,l)", "reference", "0,1,0", 1)
        self.make_row("u (h,k,l)", "u", "0,0,-1", 2)
        self.make_row("lamella (h,k,l)", "lamella", "0,1,0", 3)
        self.make_row("polishing angle (deg.)", "angle", "1.5", 4)
        
        self.axis_photo = tk.PhotoImage(file="orientation.png")
        self.axis_label = tk.Label(self.root, image=self.axis_photo)
        self.axis_label.grid(row=1, column=2, rowspan=4, padx=10, pady=5)




        button_frame = tk.Frame(root)
        button_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        

        tk.Button(button_frame, text="Calculate angles", command=self.run_calc).grid(row=0, column=1)
        
        tk.Button(button_frame, text="Show unit cell", command=self.show_unit_cell).grid(row=0, column=0, sticky="e", padx=10)
        
        tk.Button(button_frame, text="Open crystal movie", command=self.open_crystal_movie).grid(row=0, column=2, sticky="w", padx=10)
        
        self.output = scrolledtext.ScrolledText(root, width=90, height=25)
        self.unit_cell_text = None
        self.output.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

    def make_row(self, label, attr, default, row):

        tk.Label(self.root, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)

        entry = tk.Entry(self.root, width=30)

        entry.insert(0, default)

        entry.grid(row=row, column=1, padx=5, pady=5)

        setattr(self, attr, entry)

    def browse_file(self):

        path = filedialog.askdirectory(title="Select XRD data folder")

        if path:
            self.ub_file.set(path)

    def parse_vector(self, text):

        return np.array([float(x.strip()) for x in text.split(",")])
    
    def show_unit_cell(self):
        if self.unit_cell_text is None:
            messagebox.showwarning("Warning", "Please calculate first.")
            return
        messagebox.showinfo("Used unit cell", self.unit_cell_text)
        
    def load_axis_image(self):
    
        path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
    
        if not path:
            return
    
        self.axis_photo = tk.PhotoImage(file=path)
    
        self.axis_label.config(image=self.axis_photo, text="")
        
    def open_crystal_movie(self):
        movie_script = script_path("crystal movie_real.py")
        subprocess.Popen([sys.executable, movie_script])
        
    def run_calc(self):

        try:

            u = self.parse_vector(self.u.get())
            lamella = self.parse_vector(self.lamella.get())
            angle = float(self.angle.get())
            reference = self.parse_vector(self.reference.get())
            ub_file = self.ub_file.get()

            result_text, self.unit_cell_text = calculate_all(u, lamella, angle, reference, ub_file)

            self.output.delete("1.0", tk.END)

            self.output.insert(tk.END, result_text)

        except Exception as e:

            messagebox.showerror("Error", str(e))

if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.mainloop()