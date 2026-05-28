import os
import re
import glob
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, 
                             QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter


class CrystalMovieViewer(QWidget):
    def __init__(self, ops, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Zoom Boundaries
        self.zoom_factor = 1.0  # Set initial tracker to baseline
        self.min_zoom = 0.8
        self.max_zoom = 5.0

        # State Initialization
        self.ops = ops
        self.xrd_folder = None
        self.movie_folder = None
        self.jpr_file = None
        self.movie_name = None
        self.image_dict = None
        self.idx = 0
        self.frame_cache = {}
        self.max_cache_size = 30
        self.wheel_busy = False
        self.measure_points = []
        self.dragging_point = None
        self.measure_angle = None
        
        # Tracking mouse delta properties to distinguish clicks from drags
        self.mouse_press_pos = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        self.status_label = QLabel("Load JPR movie", self)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Critical for the .qss file configuration to find it
        self.status_label.setObjectName("status_label") 
        
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Initialize the Graphics View Framework Pipeline
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        
        # Hide the physical scrollbars completely
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set default behavior to Hand Drag Panning
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Redirect mouse movements through event filters
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        main_layout.addWidget(self.view, stretch=1)
        self.setLayout(main_layout)

    def load_movie(self, jpr_file):
        self.jpr_file = jpr_file
        self.movie_folder = os.path.dirname(jpr_file)
        self.movie_name = os.path.splitext(os.path.basename(jpr_file))[0]
        self.xrd_folder = os.path.dirname(self.movie_folder)
        self.ops.read_ub_from_xrd_folder(self.xrd_folder)
        self.ops.read_jpr_angle_file(jpr_file)
        self.image_dict = self.ops.find_images_for_jpr(self.movie_folder, self.movie_name)

        self.idx = 0
        self.frame_cache = {}
        
        # Reset view camera dimensions back to baseline matrices
        self.reset_zoom()
        self.is_initial_load = True

        self.draw_current_frame()
        self.preload_nearby_frames()

        # Steal keyboard/mouse focus instantly so drag panning works without a pre-click
        self.view.setFocus(Qt.OtherFocusReason)

        self.status_label.setText(
            f"Movie: {self.movie_name} | Frames: {len(self.ops.frames)} | UB: {os.path.basename(self.ops.log_file)}"
        )

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.view.resetTransform()

    def enterEvent(self, event):
        """Automatically focuses the canvas when hover-entering to make panning instant."""
        if self.ops.frames is not None:
            self.view.setFocus(Qt.MouseFocusReason)
        super().enterEvent(event)

    def eventFilter(self, source, event):
        if source == self.view.viewport() and self.ops.frames is not None:
            if event.type() == event.MouseButtonPress:
                self.handle_mouse_press(event)
                return False

            elif event.type() == event.MouseMove:
                self.handle_mouse_move(event)
                if self.dragging_point is not None:
                    return True 
                return False

            elif event.type() == event.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False

            elif event.type() == event.Wheel:
                self.handle_wheel(event)
                return True

        return super().eventFilter(source, event)

    def handle_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_press_pos = event.pos()

            scene_pos = self.view.mapToScene(event.pos())
            x, y = scene_pos.x(), scene_pos.y()
            for i, (px, py) in enumerate(self.measure_points):
                if (x - px)**2 + (y - py)**2 < 14**2: 
                    self.view.setDragMode(QGraphicsView.NoDrag)  # Lock panning for precise drag
                    self.dragging_point = i
                    return

    def handle_mouse_move(self, event):
        if self.dragging_point is not None:
            scene_pos = self.view.mapToScene(event.pos())
            self.measure_points[self.dragging_point] = (scene_pos.x(), scene_pos.y())
            self.frame_cache.pop(self.idx, None)
            self.draw_current_frame()

    def handle_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            if self.dragging_point is not None:
                self.dragging_point = None
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                return

            if self.mouse_press_pos is not None:
                travel_distance = (event.pos() - self.mouse_press_pos).manhattanLength()

                # If it's a quick, stationary tap, generate a measurement node on release
                if travel_distance < 5:
                    scene_pos = self.view.mapToScene(event.pos())
                    x, y = scene_pos.x(), scene_pos.y()

                    if len(self.measure_points) < 2:
                        self.measure_points.append((x, y))
                    else:
                        self.measure_points = [(x, y)]

                    self.frame_cache.pop(self.idx, None)
                    self.draw_current_frame()

            self.mouse_press_pos = None
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        elif event.button() == Qt.RightButton:
            self.clear_measurement()

    def handle_wheel(self, event):
        delta = event.angleDelta().y()
        
        # System A: View Matrix Scaling (Ctrl held down)
        if event.modifiers() == Qt.ControlModifier:
            zoom_step = 1.15
            if delta > 0:
                new_zoom = self.zoom_factor * zoom_step
            else:
                new_zoom = self.zoom_factor / zoom_step

            if self.min_zoom <= new_zoom <= self.max_zoom:
                self.zoom_factor = new_zoom
                if delta > 0:
                    self.view.scale(zoom_step, zoom_step)
                else:
                    self.view.scale(1 / zoom_step, 1 / zoom_step)
            return

        elif event.modifiers() == Qt.ShiftModifier:
            if delta < 0:
                self.next_frame(jump=True)
            else:
                self.prev_frame(jump=True)
            return

        # System B: Native Progression Frame switching
        if self.wheel_busy:
            return
        self.wheel_busy = True
        
        if delta < 0:
            self.next_frame()
        else:
            self.prev_frame()

        QTimer.singleShot(30, self.release_wheel)

    def release_wheel(self):
        self.wheel_busy = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        # FIX: Only run fitInView if the user hasn't actively applied a manual custom zoom.
        # This keeps top label updates from resetting your current zoom/pan state!
        if getattr(self, 'zoom_factor', 1.0) == 1.0:
            items = self.scene.items()
            if items:
                self.view.fitInView(items[0], Qt.KeepAspectRatio)

    def pil_to_pixmap(self, img):
        im = img.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGBA8888)
        return QPixmap.fromImage(qim)

    def draw_current_frame(self):
        if self.ops.frames is None:
            return

        angles = self.ops.frames[self.idx]
        frame_number = angles["frame"]

        if frame_number not in self.image_dict:
            QMessageBox.critical(self, "Error", f"No image found for frame {frame_number}")
            return

        cache_key = self.idx

        if cache_key in self.frame_cache:
            pixmap = self.frame_cache[cache_key]
        else:
            img_path = self.image_dict[frame_number]
            img = Image.open(img_path)

            img = self.ops.draw_axes_on_image(img, self.ops.UB_matrix, angles)

            # Keep a reliable coordinate space size so frame transitions don't alter the scene bounding box
            img.thumbnail((1200, 800))
            img = self.draw_measurement_overlay(img)

            pixmap = self.pil_to_pixmap(img)
            self.frame_cache[cache_key] = pixmap

            if len(self.frame_cache) > self.max_cache_size:
                oldest_key = sorted(self.frame_cache.keys())[0]
                del self.frame_cache[oldest_key]

        # Save current transformation matrices (Your exact Zoom level + Pan location)
        old_transform = self.view.transform()

        self.scene.clear()
        pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        # If loading a brand new movie file, force it to fit perfectly inside the application frame bounds
        if getattr(self, 'is_initial_load', False):
            self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            self.is_initial_load = False
            self.zoom_factor = 1.0
        else:
            # Forcefully re-apply the zoom/pan perspective coordinates to the fresh image frame swap
            self.view.setTransform(old_transform)

        img_path = self.image_dict[frame_number]
        self.status_label.setText(
            f"{self.idx + 1}/{len(self.ops.frames)} | {os.path.basename(img_path)} | "
            f"o={angles['omega']:.2f}, t={angles['theta']:.2f}, "
            f"k={angles['kappa']:.2f}, p={angles['phi']:.2f}"
        )

    def next_frame(self, jump=False):
        if self.ops.frames is None: 
            return
        self.measure_points = []
        self.measure_angle = None
        if jump:
            self.idx = (self.idx + 10) % len(self.ops.frames)
        else:
            self.idx = (self.idx + 1) % len(self.ops.frames)
        self.draw_current_frame()

    def prev_frame(self, jump=False):
        if self.ops.frames is None: 
            return
        self.measure_points = []
        self.measure_angle = None
        if jump:
            self.idx = (self.idx - 10) % len(self.ops.frames)
        else:
            self.idx = (self.idx - 1) % len(self.ops.frames)
        self.draw_current_frame()

    def clear_measurement(self):
        self.measure_points = []
        self.dragging_point = None
        self.frame_cache.pop(self.idx, None)
        self.draw_current_frame()

    def draw_measurement_overlay(self, img):
        if not self.measure_points:
            return img
        draw = ImageDraw.Draw(img)
        for x, y in self.measure_points:
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="yellow")
        if len(self.measure_points) == 2:
            x1, y1 = self.measure_points[0]
            x2, y2 = self.measure_points[1]
            draw.line([x1, y1, x2, y2], fill="yellow", width=2)
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(-dy, dx))
            font = ImageFont.load_default(size=30)
            text = f"Angle : {angle:.2f} deg"
            draw.text((x2 + 10, y2 + 10), text, fill="yellow", font=font)
        return img

    def preload_nearby_frames(self):
        if self.ops.frames is None or self.ops.UB_matrix is None:
            return
        n = len(self.ops.frames)
        preload_range = 10
        for offset in range(-preload_range, preload_range + 1):
            i = (self.idx + offset) % n
            if i in self.frame_cache:
                continue
            angles = self.ops.frames[i]
            frame_number = angles["frame"]
            if frame_number not in self.image_dict:
                continue
            try:
                img_path = self.image_dict[frame_number]
                img = Image.open(img_path)

                img = self.ops.draw_axes_on_image(img, self.ops.UB_matrix, angles)
                img.thumbnail((1100, 800))
                img = self.draw_measurement_overlay(img)
                self.frame_cache[i] = self.pil_to_pixmap(img)
            except:
                pass
        keep = set((self.idx + offset) % n for offset in range(-20, 21))
        for key in list(self.frame_cache.keys()):
            if key not in keep:
                del self.frame_cache[key]
