Traffic Violation Detection System
Overview
This project is a Traffic Violation Detection System that uses computer vision to detect traffic light violations, white line crossings, and license plate recognition. The system is built using Python, Flask, and OpenCV, and it integrates with a MySQL database to store violation records.

![image](https://github.com/user-attachments/assets/796f5b0a-2dbb-4a89-9f71-712e2dbecf5c)


Features
Real-Time Video Processing: Streams video and processes frames in real-time to detect traffic violations.
License Plate Recognition: Detects and extracts license plates using OpenCV and performs OCR (Optical Character Recognition) using Tesseract.
Traffic Light Detection: Identifies traffic light signals (red, yellow, green) to determine when a violation occurs.
White Line Detection: Detects if a vehicle crosses a designated white line when the traffic light is red.
Violation Recording: Stores detected violations in a MySQL database.
Challan/Ticket Issuing: Allows police officers to issue challans for detected violations via a user-friendly web interface.
Installation
Prerequisites
Python 3.x
MySQL server
Tesseract-OCR
Step-by-Step Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/traffic-violation-detection.git
cd traffic-violation-detection
Set Up a Virtual Environment (Optional but recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the Required Dependencies

bash
Copy code
pip install -r requirements.txt
Set Up MySQL Database

Create a MySQL database named traffic_violations_db.
Update the DB_HOST, DB_USER, DB_PASSWORD, and DB_NAME constants in app.py with your MySQL credentials.
sql
Copy code
CREATE DATABASE traffic_violations_db;
Download and Install Tesseract-OCR

Tesseract Installation Guide
Update Tesseract Path

In app.py, ensure the pytesseract.pytesseract.tesseract_cmd variable points to the correct path where Tesseract is installed.
python
Copy code
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
Run the Application

bash
Copy code
python app.py
Access the Web Interface

Open your web browser and go to http://127.0.0.1:5000/ to access the application.
Usage
Live Video Feed: View the live video feed and see real-time detection of traffic violations.
Detected License Plates: View a list of detected license plates along with the violation details.
Issue Tickets: Use the web interface to manually issue tickets for detected violations.
File Structure
app.py: The main application file containing the Flask backend and OpenCV processing logic.
static/: Contains static assets like CSS and JavaScript files.
templates/: Contains HTML templates for the web interface.
requirements.txt: Lists all the Python dependencies needed to run the project.
README.md: This documentation file.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
OpenCV
Tesseract-OCR
Flask
This README.md file provides a comprehensive guide for anyone who wants to understand, install, and use your Traffic Violation Detection System. Adjust any part of it as necessary to better fit your project's specific needs and structure.
