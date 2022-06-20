from flask import Flask,render_template,request,json,jsonify,session,redirect,send_file,url_for,flash
import os
from werkzeug.utils import secure_filename
import SRGAN_module as srgan
import systemcheck

app=Flask(__name__)
app.secret_key="secure"
app.config['UPLOAD_FOLDER'] = str(os.getcwd())+'/static/uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','bmp'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["post","get"])
def first_page():
    if request.method=="POST":
        global image_name,image_data

        file = request.files['file']
        is_degrade = request.form.get("degrade")

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)


        if file and allowed_file(file.filename) :

            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if is_degrade=="on":
                in_file_add = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                res_file_add = os.path.join(app.config['UPLOAD_FOLDER'], "srgan_degraded_image.jpg")
                srgan.predict_degrade(in_file_add,res_file_add)
                result = "srgan_degraded_image.jpg"
                return render_template("data_page.html", result = result)
            elif is_degrade == None:
                in_file_add = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                res_file_add = os.path.join(app.config['UPLOAD_FOLDER'], "srgan_normal_image.jpg")
                srgan.predict_normal(in_file_add,res_file_add)
                result = "srgan_normal_image.jpg"
                return render_template("data_page.html", result = result)
            else:
                return redirect(request.url)

            
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename))



app.run(debug=True)