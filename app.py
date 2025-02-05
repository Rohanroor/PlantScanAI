from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import io
import base64

app = Flask(__name__)

# Set the upload folder (make sure it exists)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create if it doesn't exist

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)  # No file uploaded

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)  # Empty file

        if file and allowed_file(file.filename):
            # Option 1: Save the file (useful for further processing later)
            # filename = secure_filename(file.filename) # Use secure_filename in production
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # uploaded_image = url_for('display_image', filename=filename) # For displaying from file system

            # Option 2 (Recommended for just displaying): Encode to base64
            image_bytes = file.read()
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')
            uploaded_image = f"data:{file.content_type};base64,{encoded_string}"

            return render_template('index.html', uploaded_image=uploaded_image)  # Pass to template

    return render_template('index.html', uploaded_image=None)  # Initial page load


# (Optional) Serve files from the upload folder (if you choose Option 1 above)
# @app.route('/uploads/<filename>')
# def display_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production