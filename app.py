import cv2
import time
import threading
from flask import Response, Flask, render_template, request
import frame
from inference import infer_img
tracker = frame.tracker

# Image frame sent to the Flask object

video_frame = frame.video_frame

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# Create the Flask object for the application
app = Flask(__name__)

def captureFrames():
    global thread_lock

    # Video capturing from OpenCV
    video_capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)

    while True and video_capture.isOpened():
        return_key, frame = video_capture.read()
        if not return_key:
            break

        frame = infer_img(frame)
        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        with thread_lock:
            video_frame.img = frame.copy()
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    video_capture.release()
        
def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            if video_frame.img is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame.img)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(encodeFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""

    #get data from the webpage
    form_data = request.form
    #Update the tracker hyperparameters if user inputs something
    if form_data["yolo_confidence"]:
        tracker.yolo_confidence = int(form_data["yolo_confidence"])
    if form_data["min_hits"]:
        tracker.min_hits = int(form_data["min_hits"])
    if form_data["max_age"]:
        tracker.max_age = int(form_data["max_age"])

    return render_template('index.html', conf = tracker.yolo_confidence, min_hits = tracker.min_hits, max_age = tracker.max_age)


# check to see if this is the main thread of execution
if __name__ == '__main__':

    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run("0.0.0.0", port="8000")