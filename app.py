# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper


def overlaytext(res):
    res_str = res
    save_dir_start = res_str.find("save_dir: ") + len("save_dir: ")
    save_dir_end = res_str.find("\n", save_dir_start)
    save_dir = res_str[save_dir_start+1:save_dir_end-1].strip()


    file_path = f'{save_dir}/labels.txt'



    print("Save dir: ", save_dir)
    print("File path: ", file_path)

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Initialize counts
        counts = {"Investigating": 0, "Lying": 0, "Standing": 0, "Sleeping": 0}

        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)

            # Process the class_id and update counts accordingly
            if class_id == 0:
                counts["Investigating"] += 1
            elif class_id == 1:
                counts["Lying"] += 1
            elif class_id == 2:
                counts["Standing"] += 1
            elif class_id == 3:
                counts["Sleeping"] += 1

    # # Overlay object counts on the video frame
    # overlay_text = f'Investigating: {counts["Investigating"]}\nLying: {counts["Lying"]}\nStanding: {counts["Standing"]}\nSleeping: {counts["Sleeping"]}'
    
    # # Add overlay text to the top-right corner with blue color
    # cv2.putText(image, overlay_text, (image.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    overlay_text_investigating = f'Investigating: {counts["Investigating"]}'
    overlay_text_lying = f'Lying: {counts["Lying"]}'
    overlay_text_standing = f'Standing: {counts["Standing"]}'
    overlay_text_sleeping = f'Sleeping: {counts["Sleeping"]}'

    # Get the size of the overlay text
    text_size = cv2.getTextSize(overlay_text_investigating, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

    # Calculate the position dynamically based on the height of the text
    text_position = (image.shape[1] - text_size[0] - 10, 30 + text_size[1])

    # Add overlay text to the image with blue color
    cv2.putText(image, overlay_text_investigating, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, overlay_text_lying, (text_position[0], text_position[1] + text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, overlay_text_standing, (text_position[0], text_position[1] + 2 * text_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, overlay_text_sleeping, (text_position[0], text_position[1] + 3 * text_size[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")


model_type = 'Detection'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

model_path = Path(settings.DETECTION_MODEL)
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence,save_txt = True,project="report", name="sub"
                                    )
                #overlaytext(str(res[0]))
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)



else:
    st.error("Please select a valid source type!")

# helper.play_stored_video(0.85, model)