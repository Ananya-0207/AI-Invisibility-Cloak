# ===================== SSL FIX =====================
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ===================== IMPORTS =====================
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import time

# ===================== LOAD DEEPLAB =====================
print("[INFO] Loading DeepLabV3 model...")

model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'deeplabv3_mobilenet_v3_large',
    pretrained=True
)
model.eval()
model = model.to("cpu")
torch.set_num_threads(4)

print("[INFO] Model loaded successfully")

# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ===================== VIDEO WRITER =====================
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    'cloak_output.avi',
    fourcc,
    20.0,
    (640, 480)
)

print("[INFO] Output will be saved as cloak_output.avi")

# ===================== CAPTURE BACKGROUND =====================
print("[INFO] Capturing background... Move out of frame")
time.sleep(2)

bg_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if ret:
        bg_frames.append(frame)

background = np.median(bg_frames, axis=0).astype(np.uint8)
background = cv2.flip(background, 1)

print("[INFO] Background captured successfully")

# ===================== TRANSFORM =====================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===================== HSV RANGE =====================
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
kernel = np.ones((7, 7), np.uint8)

print("[INFO] Starting invisibility cloak (ESC to exit)")

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # ---------- DeepLab Segmentation ----------
    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(img)['out'][0]

    seg_mask = output.argmax(0).cpu().numpy()

    # Person class = 15
    person_mask = np.where(seg_mask == 15, 255, 0).astype(np.uint8)
    person_mask = cv2.resize(person_mask, (640, 480))

    # ---------- FACE PROTECTION (NO MEDIAPIPE) ----------
    ys, xs = np.where(person_mask == 255)

    face_mask = np.zeros_like(person_mask)

    if len(ys) > 0:
        y_min = np.min(ys)
        y_max = np.max(ys)

        face_height = int((y_max - y_min) * 0.30)  # top 30%
        face_mask[y_min:y_min + face_height, :] = 255

    # ---------- CLOTH DETECTION ----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cloth_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply person constraint
    cloth_mask = cv2.bitwise_and(cloth_mask, cloth_mask, mask=person_mask)

    # Remove face area
    cloth_mask = cv2.bitwise_and(cloth_mask, cv2.bitwise_not(face_mask))

    # Cleanup
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
    cloth_mask = cv2.GaussianBlur(cloth_mask, (21, 21), 0)
    # Soft edge feathering
    edge = cv2.Canny(cloth_mask, 50, 150)
    edge = cv2.GaussianBlur(edge, (21, 21), 0)
    cloth_mask = cv2.subtract(cloth_mask, edge)


    # ---------- BACKGROUND REPLACEMENT ----------
    inv = cv2.bitwise_not(cloth_mask)

    bg_part = cv2.bitwise_and(background, background, mask=cloth_mask)
    fg_part = cv2.bitwise_and(frame, frame, mask=inv)

    final = cv2.add(bg_part, fg_part)

    # ---------- SAVE & DISPLAY ----------
    out.write(final)

    cv2.imshow("Camera", frame)
    cv2.imshow("Person Mask", person_mask)
    cv2.imshow("Face Protected Region", face_mask)
    cv2.imshow("Cloth Mask", cloth_mask)
    cv2.imshow("Invisibility Cloak", final)

    if cv2.waitKey(1) == 27:
        print("[INFO] Exiting...")
        break

# ===================== CLEANUP =====================
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
