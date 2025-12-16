# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tempfile

# Define the base network for feature extraction.
def create_base_network(input_shape):
    from tensorflow.keras import layers, Model
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(inp, x, name="base_network")

def siamese_network(input_shape):
    from tensorflow.keras import layers, Model
    input_a = layers.Input(shape=input_shape, name="input_a")
    input_b = layers.Input(shape=input_shape, name="input_b")
    
    base_network = create_base_network(input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Compute the absolute difference between the feature vectors.
    diff = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]),
                         name="abs_diff")([processed_a, processed_b])
    
    # Fully connected layers for change detection prediction.
    x = layers.Dense(64, activation='relu')(diff)
    output = layers.Dense(1, activation='sigmoid', name="change_output")(x)
    model = Model([input_a, input_b], output, name="siamese_network")
    return model

# Set input shape for the model (e.g., 128x128 grayscale images).
input_shape = (128, 128, 1)
model = siamese_network(input_shape)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


st.title("SAR Change Detection with Siamese CNN")
st.write("Upload two SAR images (from different times) to predict whether a significant change occurred.")

# File uploader widgets for two images.
uploaded_file1 = st.file_uploader("Choose SAR image at Time 1", type=["png", "jpg", "jpeg", "tif"], key="img1")
uploaded_file2 = st.file_uploader("Choose SAR image at Time 2", type=["png", "jpg", "jpeg", "tif"], key="img2")

def preprocess_image(image, target_size=(128, 128)):
    # Convert image to grayscale.
    image = image.convert("L")
    # Resize image to target size.
    image = image.resize(target_size)
    # Convert image to numpy array and normalize to [0, 1].
    image_array = np.array(image).astype("float32") / 255.0
    # Add channel dimension.
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Open images using PIL.
    image1 = Image.open(uploaded_file1)
    image2 = Image.open(uploaded_file2)
    
    st.image([image1, image2], caption=["Old Image", "New Image"], width=300)
    
    # Preprocess images.
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    
    # Add a batch dimension.
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    if st.button("Detect Change"):
        # Get the prediction from the Siamese network.
        prediction = model.predict([img1_batch, img2_batch])
        change_score = prediction[0][0]
        st.write("Change Score (0 means no significant change, 1 means significant change):", round(change_score, 4))
        if change_score > 0.5:
            st.success("Significant change detected between the two images!")
        else:
            st.info("No significant change detected.")
        
        # -------------------------
        # Compute pixel-wise difference for highlighting.
        # Remove batch dimension.
        img1_proc = np.squeeze(img1)  # shape: (128, 128)
        img2_proc = np.squeeze(img2)
        diff_map = np.abs(img1_proc - img2_proc)
        
        st.write("Pixel-wise Difference Map:")
        st.image(diff_map, caption="Difference Map", use_column_width=True)
        
        # Allow user to adjust threshold for highlighting changes.
        threshold = st.slider("Difference threshold for highlighting", 0.0, 1.0, 0.2)
        mask = diff_map > threshold
        
        # Create a red overlay on the areas where the mask is True.
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask] = [255, 0, 0]
        
        # Convert the first image to a 3-channel (color) image.
        img1_color = np.repeat(np.expand_dims(img1_proc, axis=-1), 3, axis=-1) * 255
        img1_color = img1_color.astype(np.uint8)
        
        # Adjust transparency via slider.
        alpha = st.slider("Overlay transparency", 0.0, 1.0, 0.5)
        combined = cv2.addWeighted(img1_color, 1.0, overlay, alpha, 0)
        
        st.image(combined, caption="Time 1 Image with Highlighted Changes", use_column_width=True)
