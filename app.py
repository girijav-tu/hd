
from flask import Flask, render_template, request, Response, redirect, url_for
from keras.models import load_model 
from keras.preprocessing import image
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import threading
import cv2
import datetime, time

os.environ["FLASK_ENV"] = "development"
app = Flask(__name__, template_folder='./templates')
port = 5000
#app.secret_key = "secretkey123"

# Open a ngrok tunnel to the HTTP server
public_url = 'localhost:5000'
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url
# ... Update inbound traffic via APIs to use the public-facing ngrok URL

camera_port = 0
video = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)

@app.route("/")
def home():
    return render_template('index.html')
def gen(video):
    global out, capture, rec_frame
    while True:
        success, image = video.read()
    
        if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.jpg".format(str(now).replace(":",''))])
                cv2.imwrite(p, image)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
       # video.release()
       # cv2.destroyAllWindows()

@app.route("/video_feed")
def video_feed():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/tasks", methods=["POST", "GET"])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1               
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# to keep users uploads --> static/uploads
image_folder = os.path.join('static', 'uploads')
app.config["UPLOAD_FOLDER"] = image_folder
#ALLOWED_EXTENSIONS = {'png','jpg','jpeg'} 

# Load the model
model = tensorflow.keras.models.load_model('./keras_model.h5', compile = False)
model.summary()

# Load the Labels
with open('./labels.txt', 'r') as f:
    class_names = f.read().split('\n')
#class_names


@app.route("/", methods=["GET"])
def index():
  return render_template("index.html")
    
@app.route("/submit", methods=["POST"])
def submit():
  # predicting Images
  imagefile = request.files['imagefile']
  image_path = './static/uploads/' + imagefile.filename
  imagefile.save(image_path)

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  image = Image.open(image_path)
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  image.show()
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  #print(prediction)

  maxValue = np.amax(prediction)
  #print(maxValue)

  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = maxValue
  pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
  
  if class_name[0] == index:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  elif class_name[1] == index:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  elif class_name[2] == index:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  elif class_name[3] == index:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  elif class_name[4] == index:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  else:
      return render_template('index.html', user_image=pic, predict_text=class_name.format(imagefile.filename), msg=confidence_score)
  

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
