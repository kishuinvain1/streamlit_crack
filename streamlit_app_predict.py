import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os





def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    
    print(uploaded_file)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
	st.image(image_data)
	name = image_data.name
	path = os.path.abspath(name)
	print("abs path")
	print(path)
	#print(Path.cwd())



def predict(model):
	#return model.predict(path).json()
	return model.predict("https://kishuinvain1-streamlit-crack-streamlit-app-predict-9tmpw9.streamlit.app/~/+/media/edf9973037ae0e2650858998d807c405002cf5ca407d930e6116cda7.jpg", hosted=True).json()
	
	
def main():
	rf = Roboflow(api_key="SNxIPCnRCYWXUM9lBAp4")
	project = rf.workspace().project("fleet-crack-2-wg5qy")
	model = project.version(1).model
	st.title('Pretrained model demo')
	image = load_image()
	result = st.button('Run on image')
	if result:
		st.write('Calculating results...')
		#results = predict(model, "/home/kishore/Desktop/Verifygn_Tech/9_crack_detection_2/Fleet-Crack-2.v2i.folder/train/Crack/0000.rf.151752c1ef8868d48500e83601531186.jpg")
		results = predict(model)
		st.write(results["predictions"][0]["predictions"][0]["class"])
		st.write(results["predictions"][0]["predictions"][0]["confidence"])
		#print(results["predictions"][0]["predictions"][0]["class"])
		#print(results["predictions"][0]["predictions"][0]["confidence"])
		


    
    


if __name__ == '__main__':
    main()
