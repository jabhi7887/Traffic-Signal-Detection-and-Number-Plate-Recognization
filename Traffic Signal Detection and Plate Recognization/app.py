from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error
import pytesseract
from PIL import Image
import re
from collections import deque
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database Connection Constants
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'new_password'
DB_NAME = 'traffic_violations_db'

# Load the trained Haar Cascade
license_plate_cascade = cv2.CascadeClassifier("F:/traffic_violation_detection/haarcascade_russian_plate_number.xml")

# List to store unique penalized license plate texts
penalized_texts = []
penalized_images = {}

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Traffic Light Detection
def detect_traffic_light_color(image, rect):
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_thickness = 2

    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message = "Detected Signal Status: Stop"
        color = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message = "Detected Signal Status: Caution"
        color = 'yellow'
    else:
        text_color = (0, 255, 0)
        message = "Detected Signal Status: Go"
        color = 'green'

    cv2.putText(image, message, (15, 70), font, font_scale + 0.5, text_color, font_thickness + 1, cv2.LINE_AA)
    cv2.putText(image, 34 * '-', (10, 115), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return image, color

# White Line Detection
class LineDetector:
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)

    def detect_white_line(self, frame, color, slope1=0.03, intercept1=920, slope2=0.03, intercept2=770, slope3=-0.8, intercept3=2420):
        def get_color_code(color_name):
            color_codes = {'red': (0, 0, 255), 'green': (0, 255, 0), 'yellow': (0, 255, 255)}
            return color_codes.get(color_name.lower())

        frame_org = frame.copy()

        def line1(x):
            return slope1 * x + intercept1

        def line2(x):
            return slope2 * x + intercept2

        def line3(x):
            return slope3 * x + intercept3

        height, width, _ = frame.shape

        mask1 = frame.copy()
        for x in range(width):
            y_line = line1(x)
            mask1[int(y_line):, x] = 0

        mask2 = mask1.copy()
        for x in range(width):
            y_line = line2(x)
            mask2[:int(y_line), x] = 0

        mask3 = mask2.copy()
        for y in range(height):
            x_line = line3(y)
            mask3[y, :int(x_line)] = 0

        gray = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(blurred_gray)
        edges = cv2.Canny(gray, 30, 100)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(dilated_edges, None, iterations=1)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=160, maxLineGap=5)

        x_start = 0
        x_end = width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
                intercept = y1 - slope * x1
                y_start = int(slope * x_start + intercept)
                y_end = int(slope * x_end + intercept)
                self.y_start_queue.append(y_start)
                self.y_end_queue.append(y_end)

        avg_y_start = int(sum(self.y_start_queue) / len(self.y_start_queue)) if self.y_start_queue else 0
        avg_y_end = int(sum(self.y_end_queue) / len(self.y_end_queue)) if self.y_end_queue else 0

        line_start_ratio = 0.32
        x_start_adj = x_start + int(line_start_ratio * (x_end - x_start))
        avg_y_start_adj = avg_y_start + int(line_start_ratio * (avg_y_end - avg_y_start))

        mask = np.zeros_like(frame)
        cv2.line(mask, (x_start_adj, avg_y_start_adj), (x_end, avg_y_end), (255, 255, 255), 4)

        color_code = get_color_code(color)
        if color_code == (0, 255, 0):
            channel_indices = [1]
        elif color_code == (0, 0, 255):
            channel_indices = [2]
        elif color_code == (0, 255, 255):
            channel_indices = [1, 2]
        else:
            raise ValueError('Unsupported color')

        for channel_index in channel_indices:
            frame[mask[:, :, channel_index] == 255, channel_index] = 255

        slope_avg = (avg_y_end - avg_y_start) / (x_end - x_start + np.finfo(float).eps)
        intercept_avg = avg_y_start - slope_avg * x_start

        mask_line = np.copy(frame_org)
        for x in range(width):
            y_line = slope_avg * x + intercept_avg - 35
            mask_line[:int(y_line), x] = 0

        return frame, mask_line

# License Plate Extraction
def extract_license_plate(frame, mask_line):
    gray = cv2.cvtColor(mask_line, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    non_black_points = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(non_black_points)
    w = int(w * 0.7)
    cropped_gray = gray[y:y + h, x:x + w]
    license_plates = license_plate_cascade.detectMultiScale(cropped_gray, scaleFactor=1.07, minNeighbors=15, minSize=(20, 20))
    license_plate_images = []

    for (x_plate, y_plate, w_plate, h_plate) in license_plates:
        cv2.rectangle(frame, (x_plate + x, y_plate + y), (x_plate + x + w_plate, y_plate + y + h_plate), (0, 255, 0), 3)
        license_plate_image = cropped_gray[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]
        license_plate_images.append(license_plate_image)

    return frame, license_plate_images

# OCR on License Plates
def apply_ocr_to_image(license_plate_image):
    _, img = cv2.threshold(license_plate_image, 120, 255, cv2.THRESH_BINARY)
    pil_img = Image.fromarray(img)
    full_text = pytesseract.image_to_string(pil_img, config='--psm 6')
    return full_text.strip()

# Draw Penalized Text on Frame
def draw_penalized_text(frame):
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    font_thickness = 2
    color = (255, 255, 255)
    y_pos = 180
    cv2.putText(frame, 'Fined license plates:', (25, y_pos), font, font_scale, color, font_thickness)
    y_pos += 80
    for text in penalized_texts:
        cv2.putText(frame, '->  ' + text, (40, y_pos), font, font_scale, color, font_thickness)
        y_pos += 60

# Database Operations
def create_database_and_table(host, user, password, database):
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            cursor.execute(f"USE {database}")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS license_plates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plate_number VARCHAR(255) NOT NULL UNIQUE,
                    violation_count INT DEFAULT 1,
                    camera_id VARCHAR(255),
                    location VARCHAR(255),
                    detected_image LONGBLOB
                )
            """)
            cursor.close()
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            connection.close()

def update_database_with_violation(plate_number, host, user, password, database):
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(f"SELECT violation_count FROM license_plates WHERE plate_number='{plate_number}'")
            result = cursor.fetchone()
            if result:
                cursor.execute(f"UPDATE license_plates SET violation_count=violation_count+1 WHERE plate_number='{plate_number}'")
            else:
                cursor.execute(f"INSERT INTO license_plates (plate_number) VALUES ('{plate_number}')")
            connection.commit()
            cursor.close()
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            connection.close()

def print_all_violations(host, user, password, database):
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT plate_number, violation_count, camera_id, location FROM license_plates ORDER BY violation_count DESC")
            result = cursor.fetchall()
            cursor.close()
            return result
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            connection.close()

def clear_license_plates(host, user, password, database):
    try:
        connection = mysql.connector.connect(host=host, user=user, password=password, database=database)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("DELETE FROM license_plates")
            connection.commit()
            cursor.close()
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            connection.close()

# Stream Processed Video Frames
def generate_frames():
    video_path = "F:/traffic_violation_detection/static/traffic_video.mp4"
    cap = cv2.VideoCapture(video_path)
    detector = LineDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rect = (1700, 40, 100, 250)
        frame, color = detect_traffic_light_color(frame, rect)
        frame, mask_line = detector.detect_white_line(frame, color)

        if color == 'red':
            frame, license_plate_images = extract_license_plate(frame, mask_line)
            for license_plate_image in license_plate_images:
                text = apply_ocr_to_image(license_plate_image)
                if text is not None and re.match(r'^[A-Z]{2}\s[0-9]{3,4}$', text) and text not in penalized_texts:
                    penalized_texts.append(text)
                    _, buffer = cv2.imencode('.jpg', license_plate_image)
                    plate_image_str = base64.b64encode(buffer).decode('utf-8')
                    penalized_images[text] = plate_image_str
                    update_database_with_violation(text, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

        if penalized_texts:
            draw_penalized_text(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    violations = print_all_violations(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    return render_template('index.html', violations=violations)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_video')
def process_video():
    detected_plates = []
    for text in penalized_texts:
        detected_plates.append({
            'text': text,
            'image': penalized_images.get(text)
        })
    return jsonify(detected_plates)

@app.route('/issue_ticket', methods=['POST'])
def issue_ticket():
    data = request.json
    plate_number = data['plate_number']
    camera_id = data['camera_id']
    location = data['location']
    detected_image = data['detected_image']
    update_database_with_ticket(plate_number, camera_id, location, detected_image)
    return jsonify({'status': 'success'})

def update_database_with_ticket(plate_number, camera_id, location, detected_image):
    try:
        connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(f"UPDATE license_plates SET camera_id='{camera_id}', location='{location}', detected_image='{detected_image}' WHERE plate_number='{plate_number}'")
            connection.commit()
            cursor.close()
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            connection.close()

if __name__ == "__main__":
    create_database_and_table(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    app.run(debug=True)
