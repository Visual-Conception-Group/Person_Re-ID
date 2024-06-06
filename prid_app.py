#32,base paper , https://pubmed.ncbi.nlm.nih.gov/29474911/ , https://arxiv.org/abs/2004.06578
from base64 import encodebytes
from crypt import methods
import io
import pathlib
from time import sleep
import time
# from flask_session import Session
from flask import Flask,request,jsonify,make_response,render_template,session
from flask_cors import CORS,cross_origin
import os
import shutil
import pickle
from PIL import Image
import glob
from helper import get_frame_count_v2, get_image_response, get_second_info, get_second_info_v2, get_total_frame_count, get_video_info,get_frame_count,PersonReidentification,HandleProgress,VideoProcess, get_video_info_v2
UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__,template_folder='template',static_url_path='/static')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/prid/api/v1/login",methods = ["POST"])
def login():
    data = request.get_json()
    handle_status = HandleProgress()
    if handle_status.authorize_user(data["username"],data["password"]):
        handle_status.make_username_entry(data["username"])
        return jsonify({"login_sucess" : True})
    handle_status.conn.close()
    return jsonify({"login_sucess" : False})

@app.route("/prid/api/v1/uploadvideo",methods = ["POST"])
def upload_video():
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        username = request.form["username"]
        username_wise_fs = os.path.join(app.config['UPLOAD_FOLDER'],username)
        if not os.path.exists(username_wise_fs):
            os.makedirs(username_wise_fs)
        file = request.files["video"]
        file.save(os.path.join(username_wise_fs,"video.mp4"))
        response = jsonify({"response" : True})
    except Exception as e:
        response = jsonify({"response" : False})
    return response

@app.route("/prid/api/v1/getvideopreview",methods = ["POST"])
def get_video_preview():
    user_name = request.get_json()["username"]
    metadata = get_video_info(app.config["UPLOAD_FOLDER"],user_name)
    response = jsonify(metadata)
    return response

@app.route("/prid/api/v1/getframecount",methods = ["POST"])
def frame_count():
    username = request.get_json()["username"]
    video_process = VideoProcess("crowdhuman_yolov5m.pt","static/uploads",username)
    video_process.read_video()
    video_process.annotate_each_frame()
    count = get_frame_count(app.config["UPLOAD_FOLDER"],username)
    response = jsonify({"count" : count})
    return response

@app.route("/prid/api/v1/getsecondinfo",methods = ["POST"])
def get_second_info_api():
    data = request.get_json()
    username , second_name = data["username"] , data["second_name"]
    frame_metadata = get_second_info(app.config["UPLOAD_FOLDER"],username,second_name)
    normal_frame_image_data = get_image_response(frame_metadata["normal_frame_path"])
    response = jsonify({"normal_frame" : normal_frame_image_data,
                        })
def get_image_response(image_path):
    image_name = pathlib.Path(image_path).stem
    pil_image = Image.open(image_path,mode = 'r')
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr,format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return dict(image = encoded_img , image_name = image_name)

    return response

@app.route("/prid/api/v1/prid",methods = ["POST"])
def start_prid():
    file = request.files["query_image"]
    username_wise_fs = os.path.join(app.config['UPLOAD_FOLDER'],request.form["username"])
    image_name = os.path.join(username_wise_fs,"query_image.png")
    file.save(image_name)
    person_reid = PersonReidentification(app.config["UPLOAD_FOLDER"],request.form["username"],
                    request.form["frame_info"],image_name,k = 10)
    result = person_reid.start_prid()
    response = jsonify(result)
    return response


@app.route("/prid/api/v1/getstatus", methods = ["POST"])
def get_status():
    data = request.get_json()
    handle_status = HandleProgress()
    status = handle_status.get_status(data["username"])
    print(status)
    status = status.split('|')
    if status[1] == "None":
        value = ""
    else:
        value = status[1]
    title = status[0]
    response = jsonify({"title" : title,"value" : value})
    handle_status.conn.close()
    return response

@app.route("/prid/api/v1/analysisdone",methods = ["POST"])
def analysis_done():
    data = request.get_json()
    folder = os.path.join(app.config['UPLOAD_FOLDER'],data["username"])
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return jsonify({"success" : True})

@app.route("/prid/api/v2/uploadvideo",methods = ["POST"])
def upload_video_v2():
    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        username = request.form["username"]
        username_wise_fs = os.path.join(app.config['UPLOAD_FOLDER'],username)
        if not os.path.exists(username_wise_fs):
            os.makedirs(username_wise_fs)
        file = request.files["video"]
        file.save(os.path.join(username_wise_fs,"video.mp4"))

        for key in request.files.keys():
            if key.startswith("aux_video"):
                aux_file =request.files[key]
                aux_file.save(os.path.join(username_wise_fs,key+".mp4"))
        

        response = jsonify({"response" : True})
    except Exception as e:
        response = jsonify({"response" : False, "error": str(e)})
    return response

@app.route("/prid/api/v2/getvideopreview",methods = ["POST"])
def get_video_preview_v2():
    user_name = request.get_json()["username"]
    metadata = get_video_info_v2(app.config["UPLOAD_FOLDER"],user_name)
    response = jsonify(metadata)
    return response

@app.route("/prid/api/v2/getframecount",methods = ["POST"])
def frame_count_v2():
    try:
        username = request.get_json()["username"]
        video_process = VideoProcess("crowdhuman_yolov5m.pt","static/uploads",username)
        
        time_read_video_start = time.perf_counter()
        video_process.read_video_v2()
        time_read_video_end = time.perf_counter()
        
        time_annotate_video_start = time.perf_counter()
        video_process.annotate_each_frame_v2()
        time_annotate_video_end = time.perf_counter()

        count = get_frame_count_v2(app.config["UPLOAD_FOLDER"],username)
        tot_count = get_total_frame_count(app.config["UPLOAD_FOLDER"],username)
        
        time_benchmark = {
            "time_read_video" : round(time_read_video_end - time_read_video_start, 4),
            "time_annotate" : round(time_annotate_video_end - time_annotate_video_start, 4)
        }

        response = jsonify({"count" : count, "tot_count" : tot_count, "time_benchmark" : time_benchmark})
        return response
    except Exception as e:
        response = jsonify({"count": 0, "error": str(e)})
        return response, 500

@app.route("/prid/api/v2/prid",methods = ["POST"])
def start_prid_v2():
    try:
        file = request.files["query_image"]
        username_wise_fs = os.path.join(app.config['UPLOAD_FOLDER'],request.form["username"])
        image_name = os.path.join(username_wise_fs,"query_image.png")
        file.save(image_name)
        person_reid = PersonReidentification(app.config["UPLOAD_FOLDER"],request.form["username"],
                        request.form["frame_info"],image_name,k = 10)
        prid_result, time_benchmark = person_reid.start_prid_v2()
        response = jsonify({
            "prid_result" : prid_result,
            "time_benchmark" : time_benchmark
        })
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        return response, 500

@app.route("/prid/api/v2/getsecondinfo",methods = ["POST"])
def get_second_info_api_v2():
    data = request.get_json()
    username , second_name = data["username"] , data["second_name"]
    frame_metadata = get_second_info_v2(app.config["UPLOAD_FOLDER"], username, second_name)
    normal_frame_image_data = get_image_response(frame_metadata["normal_frame_path"])
    response = jsonify({"normal_frame" : normal_frame_image_data, "object_count" : frame_metadata["no_of_person_in_frame"]})
    return response

@app.route("/prid/api/v1/prepare_demo",methods = ["POST"])
def prepare_demo():
    try:
        data = request.get_json()
        folder = os.path.join(app.config['UPLOAD_FOLDER'],data["username"])
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        

        folder2 = "./demo_data/prid"
        # if not os.path.exists(folder):
        #     os.mkdir(folder)

        shutil.copytree(folder2, folder, dirs_exist_ok=True)
        return jsonify({"success" : True})
    except Exception as e:
        response = jsonify({"error": str(e)})
        return response, 500
    
@app.route("/prid/api/v2/getframecount_demo",methods = ["POST"])
def frame_count_demo():
    try:
        with open('static/uploads/prid/cache_framecount.pkl', 'rb') as handle:
            cache_dict = pickle.load(handle)

        response = jsonify(cache_dict)
        return response
    except Exception as e:
        response = jsonify({"count": 0, "error": str(e)})
        return response, 500
        

if __name__ == "__main__":
    app.run(debug=False)