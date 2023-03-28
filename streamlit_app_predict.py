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



def predict(model, url):
    #return model.predict(path).json()
    return model.predict(url, hosted=True).json()
	
	
def main():
    #Model api for fleet-crack trained on 25_03_23	
    rf = Roboflow(api_key="SNxIPCnRCYWXUM9lBAp4")
    project = rf.workspace().project("fleet-crack-2-wg5qy")
    model = project.version(1).model
    
    #Model api for FleetGuard trained on 27_03_23
    rf2 = Roboflow(api_key="uhDFc9G6MKjrEvbfHt6B")
    project2 = rf2.workspace().project("fleetguard")
    model2 = project2.version(1).model


# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
    
    st.title('Crack Detection Demo')
    image = load_image()
    st.write('Enter the image URL')
    url = st.text_input('URL', '')
    result = st.button('Predict')
    if result:
        st.write('Calculating results...')
        results = predict(model2, url)
        print(results)
        new_img = results['image_path']
        print(new_img)
        #results = predict(model, path)
        #cls_res = results["predictions"][0]["predictions"][0]["class"]
        #conf_res = results["predictions"][0]["predictions"][0]["confidence"]
        #if(cls_res == "Crack" and conf_res <= 0.57):
        #    cls_res = "No-Crack"

        #st.write(cls_res)
        #st.write(conf_res)
        #print(results["predictions"][0]["predictions"][0]["class"])
        #print(results["predictions"][0]["predictions"][0]["confidence"])



    
    

if __name__ == '__main__':
    main()
