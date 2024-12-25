from flask import Flask, render_template, request, redirect, url_for
import os
import model

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
SAVE_FOLDER = 'static/saves'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_FOLDER'] = SAVE_FOLDER

# Ensure the upload and save folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Clear upload and save folders
def clear_folders():
    for folder in [UPLOAD_FOLDER, SAVE_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    # Reset state by clearing folders on GET request
    if request.method == 'GET':
        clear_folders()

    uploaded_file = None
    label = "No Output"

    if request.method == 'POST':
        # Check if a file is uploaded
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "Data.jpg")
            savepath = os.path.join(app.config['SAVE_FOLDER'], "Output.jpg")
            outputpath = savepath
            file.save(filepath)

            # Call the model for prediction
            result = model.test_model(filepath, savepath)
            if result == "Pneumonia":
                outputpath = savepath
            else:
                outputpath = filepath

            # Set values to be passed to the template
            uploaded_file = outputpath
            label = result

    # Render the page with or without an uploaded file
    return render_template('index.html', uploaded_file=uploaded_file, label=label)

# About route
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
