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
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
        return path
        #print(Path.cwd())



def predict(model, path):
	#return model.predict(path).json()
	return model.predict(url, hosted=True).json()
	
	
def main():
        rf = Roboflow(api_key="SNxIPCnRCYWXUM9lBAp4")
        project = rf.workspace().project("fleet-crack-2-wg5qy")
        model = project.version(1).model
        st.title('Pretrained model demo')
        st.write('Enter the image URL')
        url = st.text_input('URL', 'http://...')
	st.write('Image URL is: ', url)
        result = st.button('Run on image')
	if result:
		st.write('Calculating results...')
		results = predict(model, url)
		#results = predict(model, path)
		st.write(results["predictions"][0]["predictions"][0]["class"])
		st.write(results["predictions"][0]["predictions"][0]["confidence"])
		#print(results["predictions"][0]["predictions"][0]["class"])
		#print(results["predictions"][0]["predictions"][0]["confidence"])
		


    
    


if __name__ == '__main__':
    main()
