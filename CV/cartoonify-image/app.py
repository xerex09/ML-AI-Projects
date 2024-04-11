import streamlit as st
from nst import stylizeImage
from PIL import Image

def main():
    st.title("Neural Style Transfer App")
    st.write("Transform your images with artistic styles!")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        content_image = Image.open(uploaded_file)
        st.image(content_image, caption='Uploaded Image', use_column_width=True)

        style_options = {
            1: "Style 1",
            2: "Style 2"
        }
        style_choice = st.selectbox("Select a style", options=list(style_options.values()))

        if st.button("Apply Style"):
            with st.spinner("Applying style..."):
                try:
                    style_id = list(style_options.keys())[list(style_options.values()).index(style_choice)]
                    stylized_image = stylizeImage(content_image, style_id)
                    show_stylized_image(stylized_image)
                except Exception as e:
                    st.error("An error occurred. Please try again.")

def show_stylized_image(image):
    img = image.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    st.image(img, caption='Stylized Image', use_column_width=True)

if __name__ == "__main__":
    main()
