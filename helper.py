from ultralytics import YOLO
import time
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np


import settings
import subprocess
import re

# Initialize lists to store counts
sitting_counts = []
standing_counts = []
investigating_counts = []
lying_counts = []



# Create initial plots
global fig, axes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))



# def update_and_display_activity_graphs(counts):
#     """
#     Updates the counts of pigs in different activities and displays line graphs.

#     Parameters:
#         counts (dict): Dictionary containing the counts of pigs in different activities.

#     Returns:
#         None
#     """
#     # global sitting_counts, standing_counts, investigating_counts, lying_counts

#     # # Update the counts lists
#     # sitting_counts.append(counts.get("Sitting", 0))
#     # standing_counts.append(counts.get("Standing", 0))
#     # investigating_counts.append(counts.get("Investigating", 0))
#     # lying_counts.append(counts.get("Lying", 0))

#     # # Create time indices for the x-axis
#     # time_indices = np.arange(len(sitting_counts))

#     # # Update the data of the existing plots
#     # axes[0, 0].clear()
#     # axes[0, 0].plot(time_indices, sitting_counts, marker='o', color='r')
#     # axes[0, 0].set_title('Sitting Counts')
#     # axes[0, 0].set_xlabel('Time')
#     # axes[0, 0].set_ylabel('Count')

#     # axes[0, 1].clear()
#     # axes[0, 1].plot(time_indices, standing_counts, marker='o', color='g')
#     # axes[0, 1].set_title('Standing Counts')
#     # axes[0, 1].set_xlabel('Time')
#     # axes[0, 1].set_ylabel('Count')

#     # axes[1, 0].clear()
#     # axes[1, 0].plot(time_indices, investigating_counts, marker='o', color='b')
#     # axes[1, 0].set_title('Investigating Counts')
#     # axes[1, 0].set_xlabel('Time')
#     # axes[1, 0].set_ylabel('Count')

#     # axes[1, 1].clear()
#     # axes[1, 1].plot(time_indices, lying_counts, marker='o', color='m')
#     # axes[1, 1].set_title('Lying Counts')
#     # axes[1, 1].set_xlabel('Time')
#     # axes[1, 1].set_ylabel('Count')


#     # st.pyplot(fig)
    

    
#     global sitting_counts, standing_counts, investigating_counts, lying_counts

#     # Update the counts lists
#     sitting_counts.append(counts.get("Investigating", 0))
#     standing_counts.append(counts.get("Standing", 0))
#     investigating_counts.append(counts.get("Lying", 0))
#     lying_counts.append(counts.get("Sleeping", 0))

#     # Create time indices for the x-axis
#     time_indices = np.arange(len(sitting_counts))

#     # Create a new figure
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))

#     # Plot the updated data
#     axes[0, 0].plot(time_indices, sitting_counts, marker='o', color='r')
#     axes[0, 0].set_title('Sitting Counts')
#     axes[0, 0].set_xlabel('Time')
#     axes[0, 0].set_ylabel('Count')

#     axes[0, 1].plot(time_indices, standing_counts, marker='o', color='g')
#     axes[0, 1].set_title('Standing Counts')
#     axes[0, 1].set_xlabel('Time')
#     axes[0, 1].set_ylabel('Count')

#     axes[1, 0].plot(time_indices, investigating_counts, marker='o', color='b')
#     axes[1, 0].set_title('Investigating Counts')
#     axes[1, 0].set_xlabel('Time')
#     axes[1, 0].set_ylabel('Count')

#     axes[1, 1].plot(time_indices, lying_counts, marker='o', color='m')
#     axes[1, 1].set_title('Lying Counts')
#     axes[1, 1].set_xlabel('Time')
#     axes[1, 1].set_ylabel('Count')

#     # Display the updated plot
#     return st.pyplot(fig)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    # display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    # is_display_tracker = True if display_tracker == 'Yes' else False
    is_display_tracker = False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None



def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            st_frame2 = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    print("Displaying detected frames called" )
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_frame2,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                    print("Displaying detected frames finished" )
                else:
                    vid_cap.release()
                    break
            st_frame2.empty()
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))







def _display_detected_frames(conf, model, st_frame,st_frame2, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    print("************kkkkkk************")
    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf,save_txt = True,project="report", name="sub")

    
    print(type(res))
    print(res)

    res_str = str(res[0])
    save_dir_start = res_str.find("save_dir: ") + len("save_dir: ")
    save_dir_end = res_str.find("\n", save_dir_start)
    save_dir = res_str[save_dir_start+1:save_dir_end-1].strip()

    path_start = res_str.find("path: ") + len("path: ")
    path_end = res_str.find("\n", path_start)
    path = res_str[path_start+1:path_end-1].strip()
    file_path = f'{save_dir}/labels/{path.replace("jpg", "txt")}'

    print("Save dir: ", save_dir)
    print("Path: ", path)
    print("File path: ", file_path)
    
    try:
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
    except Exception as e:
        print("Error reading file: ", e)
        counts = {"Investigating": 0, "Lying": 0, "Standing": 0, "Sleeping": 0}
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


    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    

    activities = list(counts.keys())
    counts_values = list(counts.values())

    fig, ax = plt.subplots()
    ax.bar(activities, counts_values, color='blue')
    ax.set_xlabel('Activity')
    ax.set_ylabel('Number of Pigs')
    ax.set_title('Pig Activity')
    ax.grid(True)

    fig.canvas.draw()
    plt.close(fig)
    graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    st_frame2.image(graph_image, caption='Pig Activity Graph')
    





def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    print("Displaying detected frames called" )
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

