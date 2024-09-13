import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import joblib
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Cache the model loading to optimize resource usage
@st.cache_resource
def load_model():
    vgg19_weights_path = 'vgg19/vgg19_weights.pth'

    # Check if the vgg19 folder exists, if not download the model
    if not os.path.exists('vgg19'):
        os.makedirs('vgg19')  # Create directory for vgg19 weights
        # Download the VGG19 pre-trained weights
        vgg19 = models.vgg19(pretrained=True)
        torch.save(vgg19.state_dict(), vgg19_weights_path)
    else:
        # Load the VGG19 model from saved weights
        vgg19 = models.vgg19()
        vgg19.load_state_dict(torch.load(vgg19_weights_path, map_location=torch.device('cpu')))
    
    # Remove the classifier part (only using the feature extractor)
    model = torch.nn.Sequential(*list(vgg19.children())[:-2])
    model.eval()
    
    return model

# Load the VGG19 model (cached)
vgg19 = load_model()

# Transformation to resize and normalize images for VGG19
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract image embeddings using VGG19
def get_image_features(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = vgg19(image).numpy().flatten()  # Flatten the output tensor
    return features

# Function to randomly select 15 images from the folder
def get_random_images(image_filenames, num_images=15):
    return random.sample(image_filenames, min(num_images, len(image_filenames)))

# Load pre-saved embeddings
embeddings = joblib.load('image-embeddings/image_embeddings.joblib')

# Get list of image filenames
image_folder = 'transform-resized-images'
image_filenames = os.listdir(image_folder)

# Streamlit app interface
def main():
    st.title("Design Similarity Finder")

    # Upload image section
    uploaded_file = st.file_uploader("Upload a design (without mock-up)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Open the uploaded image
        uploaded_image = Image.open(uploaded_file).convert('RGB')
        
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Design")
        
        # Get embeddings of the uploaded image
        uploaded_image_embedding = get_image_features(uploaded_image)
        
        # Calculate cosine similarity with all saved embeddings
        similarities = {}
        for filename, embedding in embeddings.items():
            similarity = cosine_similarity([uploaded_image_embedding], [embedding])[0][0]
            similarities[filename] = similarity
        
        # Get top 5 similar designs
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_5_similar_images = [(filename, similarity) for filename, similarity in sorted_similarities[0:5]]
        
        st.write("---")
        st.write(f"### Top 5 Similar Designs for Uploaded Design")
        
        # Display top 5 similar design in a 3-column layout
        columns = st.columns(3)
        for idx, (filename, similarity) in enumerate(top_5_similar_images):
            col = columns[idx % 3]
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path)
            with col:
                st.image(image, caption=f"{filename}\nSimilarity: {similarity*100:.2f}%", use_column_width=True)

    # Display a ruler after the top 5 similar designs section
    st.write("---")
    
    # Display random 15 designs below the top 5 similar designs
    st.write("### Designs")
    
    # Get random 15 designs from the folder
    random_images = get_random_images(image_filenames)
    
    columns = st.columns(3)  # Create 3 columns layout
    for idx, filename in enumerate(random_images):
        col = columns[idx % 3]
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path)
        with col:
            st.image(image, caption=filename, use_column_width=True)

if __name__ == "__main__":
    main()
