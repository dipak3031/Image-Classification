import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

model = keras.models.load_model(r"C:\Users\user\OneDrive\Desktop\project\model1_cifar_10epoch.h5")

def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

Feedback_data = []  # Placeholder for storing feedback data

def home():
    st.title("Home")
    st.title("Image Classification")
    st.write("Upload an image for classification.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        image = preprocess_image(uploaded_file)

        # Display the uploaded image with increased size and improved quality
        st.image(image[0], width=400, channels="RGB", use_column_width=False, clamp=False, caption="Uploaded Image")

        # Prediction button
        if st.button("Predict"):
            # Make predictions
            predictions = model.predict(image)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_class = np.argmax(predictions[0])
            st.write("Predicted Class:", class_names[predicted_class])
            st.write("Confidence:", predictions[0][predicted_class])

def about():
    st.title("About")
    image1 = Image.open(r"C:\Users\user\OneDrive\Desktop\project\download2.png")
    st.image(image1, caption="Convolutional Neural Networks", use_column_width=True)
    st.write("             A convolutional neural network (CNN or convnet) is a subset of machine learning. It is one of the various types of artificial neural networks which are used for different applications and data types. A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data.There are other types of neural networks in deep learning, but for identifying and recognizing objects, CNNs are the network architecture of choice. This makes them highly suitable for computer vision (CV) tasks and for applications where object recognition is vital, such as self-driving cars and facial recognition.")
    
    image2_path = r"C:\Users\user\OneDrive\Desktop\project\download(.jpeg"
    image2 = Image.open(image2_path)
    st.image(image2, caption="Image Classification", use_column_width=True)
    st.write("CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns to optimize the filters (or kernels) through automated learning, whereas in traditional algorithms these filters are hand-engineered. This independence from prior knowledge and human intervention in feature extraction is a major advantage.")


   

   

def contact():
    st.title("Contact Us")
    
    st.write("Please feel free to contact any of the following team members:")
    
    st.subheader("Team Member 1")
    st.write("Name: Dipak Wagh")
    st.write("Email: dipakwaghpa3031@gmail.com")
    st.write("Contact No: +919766760901")
    
    st.subheader("Team Member 2")
    st.write("Name: Kanishk Pagare")
    st.write("Email: kanishk3117@gmail.com")
    st.write("Contact No: +917083258620")
    
    st.subheader("Team Member 3")
    st.write("Name: Shubham Daundkar")
    st.write("Email: shubhamdaundkar@gmail.com")
    st.write("Contact No: +919607861297")


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "About", "Contact Us"])

    if app_mode == "Home":
        home()
    elif app_mode == "About":
        about()
    elif app_mode == "Contact Us":
        contact()
    

if __name__ == "__main__":
    main()
