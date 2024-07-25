from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import logging
from FRP_helper import set_up_doc,get_answer
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'csv'}
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        set_up_doc(file_path)
        return jsonify({"message": "File uploaded and parsed successfully", "file_path": file_path}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/parse', methods=['POST'])
def parse_file():
    data = request.json
    file_path = data.get('file_path')
    if not file_path or not os.path.isfile(file_path):
        return jsonify({"error": "Invalid file path"}), 400
    # For simplicity, we'll just read the file's content as plain text.
    set_up_doc(file_path)
    return jsonify({"doc": 'paresed'}), 200

@socketio.on('message')
def handle_message(msg):
    resp=get_answer(msg)
    # Here you would handle chat messages.
    # For demonstration purposes, we just echo the message.
    emit('response', {'question': msg, 'response': resp})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    socketio.run(app, debug=True)
