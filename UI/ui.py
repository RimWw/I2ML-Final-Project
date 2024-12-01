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

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "Data.jpg")
            savepath = os.path.join(app.config['SAVE_FOLDER'], "Output.jpg")
            outputpath = savepath
            file.save(filepath)

            result = model.test_model(filepath,savepath)
            if result == "Pneumonia":
                outputpath = savepath
            else:
                outputpath = filepath

            # Pass the file path to render_template to display the image
            return render_template('index.html', uploaded_file=outputpath, label=result)
    
    # Render the page without an uploaded file
    return render_template('index.html', uploaded_file=None, label = "No Output")

if __name__ == '__main__':
    app.run(debug=True)
