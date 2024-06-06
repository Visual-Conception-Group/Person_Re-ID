### To run the application 
1. Download the Yolo-Human object detection and place it within the same folder when prid_app.py is present 
    https://drive.google.com/file/d/1R1Z-W0XYxkmcWZZwAgMhgoGTucra4V7e/view?usp=sharing

2. Download the prid model and place it within the same folder when prid_app.py is present 
    create the  
    https://drive.google.com/file/d/1BkeUikgWyP2mCW06TOnP3JfPRvwk6cZh/view?usp=sharing

3. Install virtual environment
    pip3 install virtualenv
4. Create the virtualenv environment
    python3 -m virtualenv "virtual env name"
5. activate the virtual environment
    source "virual env name"/bin/activate
6. install required packages
    pip3 install -r requirements.txt
8. To run the app
    python3 prid_app.py

TO deploy this application in Nginx server refer this link
    https://medium.com/swlh/deploy-flask-applications-with-uwsgi-and-nginx-on-ubuntu-18-04-2a47f378c3d2