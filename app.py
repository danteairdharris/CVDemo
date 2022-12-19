import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def jj_process(image):
    global stage
    global count
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # Draw the annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
     # Extract landmarks

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        

        # Determining Jumping Jack Completeness
        hands_together = r_wrist[0] > r_shoulder[0] and l_wrist[0] < l_shoulder[0]
        hands_above = r_wrist[1] < r_shoulder[1] and l_wrist[1] < l_shoulder[1]
        feet_together = l_ankle.x < l_hip.x and r_ankle.x > r_hip.x

        if not hands_above and feet_together:
            stage = "rest"
        if stage == "rest" and hands_together and hands_above and not feet_together:
            stage = "complete"
            count += 1

            
    except:
        pass
    
    # Render counter
    # Setup status box
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(count), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
    return image


def bc_process(image):
    global stage
    global count
    global side
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # Draw the annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
     # Extract landmarks

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        

        # Calculate angle
        if side == 'Right':
            angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        else:
            angle = calculate_angle(l_shoulder, l_elbow, l_wrist)



        # Determining Bicep Curl Completeness
        if angle > 160:
            stage = "rest"
        if angle < 30 and stage =='rest':
            stage="complete"
            count +=1

            
    except:
        pass
    
    # Render counter
    # Setup status box
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

    # Rep data
    cv2.putText(image, 'REPS', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(count), 
                (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
    return image


def jj_callback(frame: av.VideoFrame)-> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image = jj_process(image)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

def bc_callback(frame: av.VideoFrame)-> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image = bc_process(image)
    return av.VideoFrame.from_ndarray(image, format="bgr24")






st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 400px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)




with st.sidebar:

    st.markdown("<h1 style='text-align: center; color: black;'>Exercises</h1>", unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: black;'>Make sure your webcam is plugged in and enabled. Choose an exercise and press 'Start' to begin tracking with the Body Pose Estimator. </h3>", unsafe_allow_html=True)

    exercise = st.radio(
            "",
            ('Jumping Jacks', 'Bicep Curls'))
    
    if exercise == 'Bicep Curls':
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: black;'>Which bicep will you be targeting?</h3>", unsafe_allow_html=True)
        side = st.radio(
                "",
                ('Left', 'Right'))
    




streaming_placeholder = st.empty()
 
    
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
    
    
if exercise == 'Jumping Jacks':
    stage = None
    count = 0

    with streaming_placeholder.container():
        webrtc_ctx = webrtc_streamer(
            key="jumping-jacks",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=jj_callback,
            async_processing=True,
        )

elif exercise == 'Bicep Curls':
    stage = None
    count = 0
    
    
    
    with streaming_placeholder.container():
        webrtc_ctx = webrtc_streamer(
            key="bicep-curls",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=bc_callback,
            async_processing=True,
        )
        



## TODO: Add logic to determine squat completeness and add squat exercise to tracker

# elif exercise == 'Squat':
#     stage = None
#     count = 0
    
    
       


 

    