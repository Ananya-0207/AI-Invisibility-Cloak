# import cv2
# import numpy as np
# import mediapipe as mp
# import time

# cap = cv2.VideoCapture(0)
# time.sleep(2)

# print("Capturing background...")
# for i in range(60):
#     ret, bg = cap.read()

# bg = np.flip(bg, axis=1)

# mp_selfie = mp.solutions.selfie_segmentation
# segment = mp_selfie.SelfieSegmentation(model_selection=1)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = np.flip(frame, axis=1)

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = segment.process(rgb)

#     mask = (results.segmentation_mask > 0.6).astype(np.uint8)

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([90, 50, 50])
#     upper_blue = np.array([130, 255, 255])

#     cloth_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     cloth_mask = cv2.bitwise_and(cloth_mask, cloth_mask, mask=mask)

#     kernel = np.ones((7,7), np.uint8)
#     cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)
#     cloth_mask = cv2.GaussianBlur(cloth_mask, (15,15), 0)

#     inverse = cv2.bitwise_not(cloth_mask)

#     res1 = cv2.bitwise_and(bg, bg, mask=cloth_mask)
#     res2 = cv2.bitwise_and(frame, frame, mask=inverse)

#     final = cv2.add(res1, res2)

#     # WINDOW 1 → Raw Camera
#     cv2.imshow("Live Camera", frame)

#     # WINDOW 2 → Camouflage Output
#     cv2.imshow("Invisibility Cloak", final)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# SSL FIX
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
import numpy as np
import torchvision.transforms as T
import time

print("[INFO] Loading DeepLab model...")
model = torch.hub.load(
    'pytorch/vision:v0.10.0',
    'deeplabv3_mobilenet_v3_large',
    pretrained=True
)
model.eval()
torch.set_num_threads(4)
print("[INFO] Model loaded.")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cloak_output.avi', fourcc, 20.0, (640,480))

# Capture background
print("[INFO] Capturing background. Please move away...")
time.sleep(2)

bg_frames = []
for _ in range(60):
    ret, frame = cap.read()
    if ret:
        bg_frames.append(frame)

background = np.median(bg_frames, axis=0).astype(np.uint8)
background = cv2.flip(background, 1)
print("[INFO] Background captured.")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Blue cloth range
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
kernel = np.ones((5,5), np.uint8)

print("[INFO] Starting camouflage...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # DeepLab person segmentation
    img = transform(frame).unsqueeze(0)
    with torch.no_grad():
        output = model(img)['out'][0]

    mask = output.argmax(0).cpu().numpy()
    person_mask = np.where(mask == 15, 255, 0).astype(np.uint8)
    person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))

    # Cloth detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cloth_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cloth_mask = cv2.bitwise_and(cloth_mask, cloth_mask, mask=person_mask)
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cloth_mask = cv2.GaussianBlur(cloth_mask, (15,15), 0)

    inv_mask = cv2.bitwise_not(cloth_mask)

    bg_part = cv2.bitwise_and(background, background, mask=cloth_mask)
    fg_part = cv2.bitwise_and(frame, frame, mask=inv_mask)

    final = cv2.add(bg_part, fg_part)

    out.write(final)

    cv2.imshow("Camera", frame)
    cv2.imshow("Cloth Mask", cloth_mask)
    cv2.imshow("Invisibility Cloak", final)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
