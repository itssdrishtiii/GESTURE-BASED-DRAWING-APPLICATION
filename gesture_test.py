import cv2
import numpy as np
import mediapipe as mp
from math import hypot
import datetime
import os  # for showing save path

# Toolbar and popup state
toolbar_height = 50
toolbar_width = 50
icon_width = 60
button_labels = ['Color', 'Shapes', 'Erase', 'Save', 'Undo', 'Redo']
selected_tool = None
button_positions = []
color_popup = False
shape_popup = False
draw_color = (0, 0, 255)  # default red
draw_shape = None  # 'line', 'rect', 'circle'
def draw_toolbar(canvas):
    global button_positions

    h, w, _ = canvas.shape
    y_toolbar = h - toolbar_height
    button_positions.clear()

    # Draw toolbar background
    cv2.rectangle(canvas, (0, y_toolbar), (w, h), (245, 245, 245), -1)

    # Draw buttons
    for i, label in enumerate(button_labels):
        x1 = i * icon_width
        x2 = x1 + icon_width
        button_positions.append((x1, y_toolbar, x2, h))
        cv2.rectangle(canvas, (x1, y_toolbar), (x2, h), (220, 220, 220), 1)

    y_center = y_toolbar + toolbar_height // 2

    # Icons:
    cv2.circle(canvas, (30, y_center), 10, (0, 0, 255), -1)  # Color
    pts = np.array([[90, y_center+10], [75, y_center-10], [105, y_center-10]], np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2)  # Shape
    cv2.rectangle(canvas, (125, y_center-10), (145, y_center+10), (0, 0, 0), 2)  # Erase
    cv2.line(canvas, (125, y_center+10), (145, y_center-10), (0, 0, 0), 2)
    cv2.rectangle(canvas, (185, y_center-10), (205, y_center+10), (0, 0, 0), 2)  # Save
    cv2.rectangle(canvas, (190, y_center-5), (200, y_center), (0, 0, 0), -1)
    cv2.arrowedLine(canvas, (245, y_center), (225, y_center), (0, 0, 0), 2, tipLength=0.4)  # Undo
    cv2.arrowedLine(canvas, (265, y_center), (285, y_center), (0, 0, 0), 2, tipLength=0.4)  # Redo

    draw_color_popup(canvas)
    draw_shape_popup(canvas)
def draw_color_popup(canvas):
    global color_popup
    if not color_popup:
        return

    colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i, col in enumerate(colors):
        x = 20 + i * 60
        y = canvas.shape[0] - toolbar_height - 60
        cv2.rectangle(canvas, (x, y), (x + 40, y + 40), col, -1)
        cv2.rectangle(canvas, (x, y), (x + 40, y + 40), (0, 0, 0), 2)

def draw_shape_popup(canvas):
    global shape_popup
    if not shape_popup:
        return

    shapes = ['Line', 'Rect', 'Circle']
    for i, name in enumerate(shapes):
        x = 100 + i * 100
        y = canvas.shape[0] - toolbar_height - 50
        cv2.rectangle(canvas, (x, y), (x + 80, y + 40), (230, 230, 230), -1)
        cv2.putText(canvas, name, (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
def handle_toolbar_click(event, x, y, flags, param):
    global selected_tool, draw_color, color_popup, shape_popup, draw_shape

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check popup click first
        if color_popup:
            for i, col in enumerate([(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]):
                if 20 + i*60 <= x <= 60 + i*60 and canvas.shape[0] - toolbar_height - 60 <= y <= canvas.shape[0] - toolbar_height - 20:
                    draw_color = col
                    color_popup = False
                    return

        elif shape_popup:
            for i, name in enumerate(['line', 'rect', 'circle']):
                if 100 + i*100 <= x <= 180 + i*100 and canvas.shape[0] - toolbar_height - 50 <= y <= canvas.shape[0] - toolbar_height - 10:
                    draw_shape = name
                    shape_popup = False
                    return

        # Normal toolbar buttons
        if y >= canvas.shape[0] - toolbar_height:
            for i, (x1, y1, x2, y2) in enumerate(button_positions):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_tool = button_labels[i]
                    print(f"Selected tool: {selected_tool}")
                    if selected_tool == 'Color':
                        color_popup = True
                        shape_popup = False
                    elif selected_tool == 'Shapes':
                        shape_popup = True
                        color_popup = False
                    elif selected_tool == 'Erase':
                        draw_color = (255, 255, 255)
                        color_popup = shape_popup = False
                    elif selected_tool == 'Save':
                        filename = f"drawing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(filename, canvas)
                        print("Saved to:", os.path.abspath(filename))
                        color_popup = shape_popup = False
                    else:
                        color_popup = shape_popup = False


# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # white canvas

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

drawing = False
draw_color = (0, 0, 255)
prev_x, prev_y = None, None
alpha = 0.2  # smoothing factor
smooth_x, smooth_y = 0, 0
cv2.namedWindow("Gesture Drawing App")  # Create window before assigning callback
cv2.setMouseCallback("Gesture Drawing App", handle_toolbar_click)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    display_canvas = canvas.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip positions
            x1, y1 = int(lm[4].x * w), int(lm[4].y * h)   # Thumb tip
            x2, y2 = int(lm[8].x * w), int(lm[8].y * h)   # Index tip

            distance = hypot(x2 - x1, y2 - y1)

            # Gesture: Clear canvas on pinch
            # Gesture: Stop drawing on pinch
            if distance < 10:
             drawing = False
             prev_x, prev_y = None, None
            cv2.putText(frame, "Drawing Stopped (Pinch)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Drawing logic with line and smoothing
            if drawing:
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x2, y2
                    smooth_x, smooth_y = x2, y2

                # Apply EMA smoothing
                smooth_x = int(alpha * x2 + (1 - alpha) * smooth_x)
                smooth_y = int(alpha * y2 + (1 - alpha) * smooth_y)

                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, 5)
                prev_x, prev_y = smooth_x, smooth_y
            else:
                prev_x, prev_y = None, None

            # Draw fingertip on webcam view
                    # Create a copy of canvas just for display
                display_canvas = canvas.copy()

# Draw a non-intrusive + cursor
                cv2.line(display_canvas, (x2 - 10, y2), (x2 + 10, y2), (0, 255, 255), 2)
                cv2.line(display_canvas, (x2, y2 - 10), (x2, y2 + 10), (0, 255, 255), 2)

                 

    # Insert camera preview in bottom-right corner
    cam_small = cv2.resize(frame, (240, 160))
   # Adjust camera preview to appear above the toolbar
    cam_x = canvas.shape[1] - 240
    cam_y = canvas.shape[0] - 160 - toolbar_height
    canvas[cam_y:cam_y+160, cam_x:cam_x+240] = cam_small


    # Display final result
    draw_toolbar(canvas)
    cv2.imshow("Gesture Drawing App", display_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        drawing = not drawing
        prev_x, prev_y = None, None
    elif key == ord('s'):
        filename = f"drawing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved drawing as {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
