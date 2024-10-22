import cv2
import numpy as np

#Creating a function that calculates mean of the colors in circle
def calculate_mean_color(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    mean_color = cv2.mean(image, mask=mask)
    return mean_color[:3]  

#setting threshold when it changes
def color_change_exceeds_threshold(initial_mean, current_mean, threshold=85):
    color_difference = np.linalg.norm(np.array(initial_mean) - np.array(current_mean))
    return color_difference > threshold 

cap = cv2.VideoCapture('video.mp4')

#Capturing the first frame
if not cap.isOpened():
    print("Error opening video stream or file")
ret, frame = cap.read()

if not ret:
    print("Failed to read the first frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()


center_coordinates = (1000, 400)
radius = 50
circle_color = (0, 0, 255)  
initial_mean_color = calculate_mean_color(frame, center_coordinates, radius)
blue_circle_counter = 0

#Looking for all frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.circle(frame, (1000, 400), 20, circle_color, -1)
    
    
    current_mean_color = calculate_mean_color(frame, center_coordinates, radius)
    if color_change_exceeds_threshold(initial_mean_color, current_mean_color):
        if circle_color != (255, 0, 0):  # If the circle was not already blue
            blue_circle_counter += 1
        circle_color = (255, 0, 0)  # Blue color in BGR format
    else:
        # Change the circle color back to red if mean color returns to initial
        circle_color = (0, 0, 255)
    initial_mean_color = current_mean_color
    text_position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    cv2.putText(frame, f'Box Count: {blue_circle_counter}', text_position, font, font_scale, font_color, line_type)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
