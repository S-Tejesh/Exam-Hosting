from bson import ObjectId
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
from pymongo import MongoClient
from bson.binary import Binary
import cv2
import mediapipe as mp
import numpy as np

from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'something'
client = MongoClient('your url here')  # Update the connection string accordingly
db = client['Integrity-Watch']  # Replace 'mydatabase' with your database name
users_collection = db['users']
exams_collection = db['exams']
stop_proctoring = False
# Load the YOLO model
yolo_model = YOLO("yolov8m.pt")

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})

        if user:
            if user['password'] == password:
                session['user_id'] = str(user['_id'])
                return redirect(url_for('home'))
            else:
                error = 'Invalid password!'
        else:
            error = 'Username not found!'

        return render_template('test.html', error=error)
    return render_template('test.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['newUsername']
    password = request.form['newPassword']

    # Check if the username already exists
    if users_collection.find_one({'username': username}):
        error= 'Username already exists!'
        return render_template('test.html', error=error)
    # Insert new user into the database
    users_collection.insert_one({'username': username, 'password': password})
    error='Signup successful!'
    return render_template('test.html', error=error)

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')
@app.route('/logout')
def logout():
    session.clear()
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return redirect(url_for('login'))

@app.route('/exams')
def exams():
    try:
        exam_id='662a9007c3e1515275594924'
        db[f'exam_{exam_id}'].insert_one({
            'user_id': session.get('user_id'),
            'wrote': False,
            'cheat_imgs': [],
            'exam_result': {},
            'cheat_score': 0
        })
        return render_template('exams.html', exams=list(exams_collection.find({'_id':ObjectId(exam_id)})))
    except Exception as e:
        print('Error retrieving exams:', e)

@app.route('/proctoring_page/<exam_id>', methods=['GET'])
def proctoring(exam_id):
    try:
        global stop_proctoring
        stop_proctoring = False
        exam_object_id = ObjectId(exam_id)
        exam_document = db['exams'].find_one({'_id': exam_object_id})
        db[f'exam_{exam_id}'].update_one(
            {'user_id': session.get('user_id')},
            {'$set': {'wrote': True}}
        )
        if exam_document is not None:
            return render_template('proctoring_page.html',exam=exam_document)
        else:
            return jsonify({'error': 'Exam not found'}), 404
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/trigger_proctoring/<exam_id>', methods=['POST'])
def trigger_proctoring(exam_id):
    # Load the YOLO model
    yolo_model = YOLO("yolov8m.pt")

    # Initialize MediaPipe Face Mesh and Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Parameters for detecting cheating
    cheating_threshold = 15  # Number of consecutive frames indicating cheating
    reset_threshold = 5  # Number of consecutive frames to reset the cheating flag
    no_face_threshold = 30  # Number of consecutive frames indicating no face detection
    multiple_face_threshold = 30  # Number of consecutive frames indicating multiple face detections

    cheating_count = 0
    reset_count = 0
    no_face_count = 0
    multiple_face_count = 0
    cheating_flag = False
    image_saved_for_cheating = False  # Flag to track whether an image has been saved for the current cheating event
    score=0
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
            mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if stop_proctoring:
                db[f'exam_{exam_id}'].update_one(
                    {'user_id': session.get('user_id')},
                    {'$set': {'cheat_score': score}}
                )
                break
            # Object detection using YOLO
            results = yolo_model.predict(frame, classes=[67])

            # Perform object detection on the frame
            objects_detected = any(r.boxes.cls.tolist() for r in results)

            if objects_detected:
                # If objects are detected, reset cheating counters
                cheating_count = 0
                reset_count = 0
                no_face_count = 0
                multiple_face_count = 0
                cheating_flag = True
                if not image_saved_for_cheating:  # Save image only if not already saved for the current event
                    image_saved_for_cheating = True
                    # Save the image where cheating is detected
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    score+=4
                    # Save the image to MongoDB as BSON
                    bson_data = Binary(img_bytes)
                    db[f'exam_{exam_id}'].update_one(
                        { 'user_id': session.get('user_id')},
                        {'$push': {'cheat_imgs': bson_data}}
                    )

            # Cheating detection using MediaPipe and OpenCV
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Detect faces
            face_results = face_detection.process(image)

            # Convert image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.detections:
                if len(face_results.detections) == 1:
                    # Convert image to RGB and process
                    ih, iw, _ = image.shape
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            face_3d = []
                            face_2d = []

                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                    x, y = int(lm.x * iw), int(lm.y * ih)

                                    # Get the 2D Coordinates
                                    face_2d.append([x, y])

                                    # Get the 3D Coordinates
                                    face_3d.append([x, y, lm.z])

                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            # The camera matrix
                            focal_length = 1 * iw
                            cam_matrix = np.array([[focal_length, 0, ih / 2],
                                                   [0, focal_length, iw / 2],
                                                   [0, 0, 1]])

                            # The distortion parameters
                            dist_matrix = np.zeros((4, 1), dtype=np.float64)

                            # Solve PnP
                            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                            # Get rotational matrix
                            rmat, jac = cv2.Rodrigues(rot_vec)

                            # Get angles
                            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                            # Get the y rotation degree
                            y = angles[1] * 360

                            # See where the user's head is looking
                            if y < -10:
                                cheating_count += 1
                                reset_count = 0  # Reset the reset counter
                                if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                    cheating_flag = True
                                    score+=1
                            elif y > 10:
                                cheating_count += 1
                                reset_count = 0  # Reset the reset counter
                                if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                    cheating_flag = True
                                    score += 1
                            else:
                                reset_count += 1
                                if reset_count >= reset_threshold:
                                    cheating_count = 0  # Reset the cheating counter
                                    cheating_flag = False
                                    image_saved_for_cheating = False  # Reset the flag
                else:
                    # Increment multiple face count and check for cheating
                    multiple_face_count += 1
                    if multiple_face_count >= multiple_face_threshold:
                        cheating_flag = True
                        multiple_face_count = 0
                        score += 2
                        # You might want to include additional actions here, such as logging or notifications
            else:
                # Increment no face count and check for cheating
                no_face_count += 1
                if no_face_count >= no_face_threshold:
                    cheating_flag = True
                    no_face_count = 0
                    score += 3
            if cheating_flag and not image_saved_for_cheating:
                _, img_encoded = cv2.imencode('.jpg', image)
                img_bytes = img_encoded.tobytes()

                # Save the image to MongoDB as BSON
                bson_data = Binary(img_bytes)
                db[f'exam_{exam_id}'].update_one(
                    {'user_id': session.get('user_id')},
                    {'$push': {'cheat_imgs': bson_data}}
                )
                image_saved_for_cheating = True  # Set the flag to indicate that the image has been saved
   # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/submit_exam/<exam_id>', methods=['POST'])
def submit_exam(exam_id):
    global stop_proctoring
    stop_proctoring = True
    exam = exams_collection.find_one({'_id': ObjectId(exam_id)})

    # Get the submitted answers from the form
    submitted_answers = {}
    for question_number, option in request.form.items():
        if question_number.startswith('question'):
            question_number = int(question_number.replace('question', ''))
            submitted_answers[question_number] = option
    # Calculate the score
    score = 0
    for i, question in enumerate(exam['questions'], 1):
        correct_answer = question['options'][int(question['answer'])]
        submitted_answer = submitted_answers.get(i)
        if submitted_answer == correct_answer:
            score += 1

    # Calculate total marks
    total_marks = len(exam['questions'])

    # Calculate percentage
    percentage = (score / total_marks) * 100

    # Prepare exam result data
    user_id = session['user_id']
    exam_result = {
        'score': score,
        'total_marks': total_marks,
        'percentage': percentage
    }
    db[f'exam_{exam_id}'].update_one({'user_id': user_id}, {'$set': {'exam_result': exam_result}})
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
