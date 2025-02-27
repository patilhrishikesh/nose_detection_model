# import cv2
# import mediapipe as mp
# import numpy as np

# # Function to warp the nose region smoothly
# def warp_nose(frame, nose_landmarks, scale_factor=1.0):
#     try:
#         # Convert landmarks to pixel coordinates
#         nose_points = np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in nose_landmarks], dtype=np.int32)

#         # Get the bounding box of the nose
#         x, y, w, h = cv2.boundingRect(nose_points)

#         # Define the center of the nose
#         center = (int(x + w / 2), int(y + h / 2))

#         # Create a mask for the nose region (single-channel)
#         mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
#         cv2.fillConvexPoly(mask, nose_points, 255)

#         # Apply affine transformation to scale the nose
#         M = cv2.getRotationMatrix2D(center, 0, scale_factor)
#         warped_nose = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

#         # Blend the warped nose region with the original frame
#         blended = cv2.seamlessClone(warped_nose, frame, mask, center, cv2.NORMAL_CLONE)

#         return blended
#     except Exception as e:
#         print(f"Error in warp_nose: {e}")
#         return frame  # Return the original frame if an error occurs

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize OpenCV video capture
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Initial scale factor
# scale_factor = 1.0

# # Text display parameters
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_color = (0, 255, 0)  # Green color
# font_thickness = 2

# try:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Convert the frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process the frame with MediaPipe Face Mesh
#         results = face_mesh.process(rgb_frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Extract nose landmarks (indices 1, 2, 4, 5, etc., depending on the model)
#                 nose_landmarks = [face_landmarks.landmark[i] for i in [1, 2, 4, 5, 6]]

#                 # Warp the nose smoothly
#                 frame = warp_nose(frame, nose_landmarks, scale_factor)

#                 # Draw nose landmarks for visualization
#                 for landmark in nose_landmarks:
#                     x = int(landmark.x * frame.shape[1])
#                     y = int(landmark.y * frame.shape[0])
#                     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#         # Display the current scale factor on the frame
#         cv2.putText(frame, f"Scale Factor: {scale_factor:.2f}", (10, 50), font, font_scale, font_color, font_thickness)

#         # Display the frame
#         cv2.imshow("Nose Detection and Scaling", frame)

#         # Adjust scale factor with keyboard input
#         key = cv2.waitKey(1)
#         if key == ord('+'):
#             scale_factor += 0.2  # Increase nose size (larger increment for noticeable effect)
#         elif key == ord('-'):
#             scale_factor -= 0.2  # Decrease nose size (larger decrement for noticeable effect)
#         elif key == ord('q'):
#             break

# except Exception as e:
#     print(f"Error during execution: {e}")

# finally:
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Resources released.")


import cv2
import mediapipe as mp
import numpy as np

# Function to stretch the nose tip vertically
def stretch_nose_tip(frame, nose_landmarks, stretch_factor=1.0):
    try:
        # Convert landmarks to pixel coordinates
        nose_points = np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in nose_landmarks], dtype=np.int32)

        # Get the bounding box of the nose
        x, y, w, h = cv2.boundingRect(nose_points)

        # Define the center of the nose tip (bottom center of the bounding box)
        center = (int(x + w / 2), int(y + h))

        # Create a mask for the nose region (single-channel, 8-bit)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, nose_points, 255)

        # Create a transformation matrix to stretch the nose tip vertically
        M = np.float32([[1, 0, 0], [0, stretch_factor, (1 - stretch_factor) * center[1]]])
        stretched_nose = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        # Blend the stretched nose region with the original frame
        blended = cv2.seamlessClone(stretched_nose, frame, mask, center, cv2.NORMAL_CLONE)

        return blended
    except Exception as e:
        print(f"Error in stretch_nose_tip: {e}")
        return frame  # Return the original frame if an error occurs

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initial stretch factor
stretch_factor = 1.0

# Text display parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)  # Green color
font_thickness = 2

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract nose landmarks (indices 1, 2, 4, 5, etc., depending on the model)
                nose_landmarks = [face_landmarks.landmark[i] for i in [1, 2, 4, 5, 6]]

                # Stretch the nose tip vertically
                frame = stretch_nose_tip(frame, nose_landmarks, stretch_factor)

                # Draw nose landmarks for visualization
                for landmark in nose_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the current stretch factor on the frame
        cv2.putText(frame, f"Stretch Factor: {stretch_factor:.2f}", (10, 50), font, font_scale, font_color, font_thickness)

        # Display the frame
        cv2.imshow("Nose Tip Stretching", frame)

        # Adjust stretch factor with keyboard input
        key = cv2.waitKey(1)
        if key == ord('+'):
            stretch_factor += 0.2  # Increase nose tip length
        elif key == ord('-'):
            stretch_factor -= 0.2  # Decrease nose tip length
        elif key == ord('q'):
            break

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")