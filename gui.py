import math
import numpy as np
import os
import glob
import subprocess
import sys
import re
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QSizePolicy
)

from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt

from ops import Operations
from movie import CrystalMovieViewer

class Gui(QWidget):
    def __init__(self):
        super().__init__()
        #try:
        #    with open("dark_theme.qss", "r") as f:
        #        self.setStyleSheet(f.read())
        #except FileNotFoundError:
        #    print("Warning: dark_theme.qss file not found. Falling back to default styles.")

        self.ops = Operations()
        self.movie_widget = CrystalMovieViewer(self.ops)
        self._file_name = None
        self._exp_folder = None
        self._curr_folder = None

        self.setWindowTitle(
            "FIB / XRD Angle Calculator"
        )

        self.setup_ui()

    # --------------------------------------------------------
    # UI setup
    # --------------------------------------------------------

    def setup_ui(self):

        #layout = QGridLayout()
        main_layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        label = QLabel("XRD experiment folder")

        self.ub_file = QLineEdit()

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        #browse_btn.clicked.connect(self.movie_widget.open_jpr_file)

        top_bar.addWidget(label)
        top_bar.addWidget(self.ub_file)
        top_bar.addWidget(browse_btn)

        main_layout.addLayout(top_bar)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        layout = QGridLayout()
        left_widget.setLayout(layout,stretch=1)

        splitter.addWidget(left_widget)
        splitter.addWidget(self.movie_widget)
        splitter.setStretchFactor(0,0)
        splitter.setStretchFactor(1,1)
    
        #splitter.setSizes([600, 700])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Input rows
        self.reference = self.make_row(
            layout,
            "reference (h,k,l)",
            "0,1,0",
            1
        )

        self.u = self.make_row(
            layout,
            "u (h,k,l)",
            "0,0,-1",
            2
        )

        self.lamella = self.make_row(
            layout,
            "lamella (h,k,l)",
            "0,1,0",
            3
        )

        self.polish_angle = self.make_row(
            layout,
            "polishing angle (deg.)",
            "1.5",
            4
        )

        # Image

        self.axis_label = QLabel()

        if os.path.exists("orientation2.png"):
            pixmap = QPixmap("orientation2.png")

            self.axis_label.setPixmap(
                pixmap.scaled(
                    200,
                    200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        layout.addWidget(self.axis_label, 1, 2, 4, 1)

        # Buttons

        button_layout = QHBoxLayout()
        self.show_btn = QPushButton("Show unit cell")
        self.show_btn.clicked.connect(self.show_unit_cell)
        self.show_btn.setEnabled(False)

        self.calc_btn = QPushButton("Calculate angles")
        self.calc_btn.clicked.connect(self.run_calc)
        self.calc_btn.setEnabled(False)

        button_layout.addWidget(self.show_btn)
        button_layout.addWidget(self.calc_btn)
        layout.addLayout(button_layout, 5, 0, 1, 3)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout.addWidget(self.output, 6, 0, 1, 3)
        self.resize(1000, 700)

    def make_row(
        self,
        layout,
        label,
        default,
        row
    ):

        layout.addWidget(QLabel(label), row, 0)
        entry = QLineEdit()
        entry.setFixedWidth(50)
        entry.setText(default)
        layout.addWidget(entry, row, 1)
        return entry

    # --------------------------------------------------------
    # GUI actions
    # --------------------------------------------------------

    def browse_file(self):
        if self._curr_folder is None:
            self._curr_folder = os.getcwd()
            
        self._file_name, _ = QFileDialog.getOpenFileName(self, 
                                                  "Select JPR movie file", 
                                                  self._curr_folder,
                                                  "JPR files (*.jpr *.JPR);;All files (*.*)"
                                                 )
        if self._file_name != '':
            self._curr_folder = os.path.dirname(self._file_name)
            self._exp_folder = os.path.dirname(self._curr_folder)
            self.ub_file.setText(self._exp_folder)
        try:
           #self.ops.read_ub_from_xrd_folder(path)
           self.movie_widget.load_movie(self._file_name)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Movie Load Warning",
                str(e)
            )

        self.show_btn.setEnabled(True)
        self.calc_btn.setEnabled(True)

    def parse_vector(self, text):
        return np.array([
            float(x.strip())
            for x in text.split(",")
        ])

    def show_unit_cell(self):
        if self.ops.unit_cell_text is None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please calculate first."
            )
            return

        QMessageBox.information(
            self,
            "Used unit cell",
            self.ops.unit_cell_text
        )

    def run_calc(self):
        try:
            reference = self.parse_vector(self.reference.text())
            u = self.parse_vector(self.u.text())
            lamella = self.parse_vector(self.lamella.text())
            angle = float(self.polish_angle.text())
            ub_file = self.ub_file.text()

            self.ops.calculate_all(
                u,
                lamella,
                angle,
                reference,
                ub_file
            )

            self.output.setPlainText(
                self.ops.result_text
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Gui()
    window.show()
    sys.exit(app.exec())
