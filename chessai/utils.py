import cv2
import numpy as np
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import pkg_resources

from chessai.config import DEFAULT_VISUALIZATION_FRAME

def open_file(path):
    """
    Open file in default application
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def extract_frontend_dist(static_folder):
    """
    Extract folder frontend/dist from package chessai
    and put it in the same static folder for serving
    """
    if os.path.exists(static_folder):
        logging.info(f"Refreshing {static_folder}...")
        shutil.rmtree(static_folder, ignore_errors=True)
    dist_folder = pkg_resources.resource_filename("chessai", "frontend-dist")
    if os.path.exists(dist_folder):
        pathlib.Path(static_folder).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dist_folder, static_folder)
    if not os.path.exists(static_folder):
        logging.warning("frontend-dist not found in package chessai")
        pathlib.Path(static_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(static_folder, "index.html"), "w") as f:
            f.write(
                "<b>frontend-dist</b> not found in package chessai. Please run: <code>bash build_frontend.sh</code>"
            )
        return


def draw_message_box(width, height, message):
    """Draws a message box with the given width, height and message"""
    message_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        message_frame,
        message,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return message_frame



def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes()


def original_frame_stream():
    while True:
        frame = None
        with globals.frame_lock:
            frame = globals.original_frame
            if frame is None:
                frame = DEFAULT_VISUALIZATION_FRAME
        encoded_frame = encode_image(frame)
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_frame) + b'\r\n')
        else:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_frame) + b'\r\n')
