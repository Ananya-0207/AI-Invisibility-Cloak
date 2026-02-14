# =====================================================
# AI Invisibility Cloak using DeepLabV3 + OpenCV
# =====================================================

# ----------- SSL Handling -----------
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ----------- Imports -----------
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import time

# ----------- Load Pretrained Segmentation Model -----------
print("[SYSTEM] Initializing DeepLabV3 model...")

segmentation_model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "deeplabv3_mobilenet_v3_large",
    pretrained=True
)

segmentation_model.eval()
segmentation_model.to("cpu")
torch.set_num_threads(4)

print("[SYSTEM] Model ready.")

# ----------- Webcam Setup -----------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------- Video Output Setup -----------
codec = cv2.VideoWriter_fourcc(*"XVID")
video_output = cv2.VideoWriter(
    "invisibility_output.avi",
    codec,
    20.0,
    (640, 480)
)

print("[SYSTEM] Recording started.")

# ----------- Background Initialization -----------
print("[SYSTEM] Capturing clean background...")
time.sleep(2)

background_collection = []

for _ in range(120):
    success, frame = camera.read()
    if success:
        background_collection.append(frame)

static_background = np.median(background_collection, axis=0).astype(np.uint8)
static_background = cv2.flip(static_background, 1)

print("[SYSTEM] Background stored.")

# ----------- Image Preprocessing Transform -----------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------- Cloak Color Range (Blue Default) -----------
blue_lower = np.array([90, 50, 50])
blue_upper = np.array([130, 255, 255])

morph_kernel = np.ones((7, 7), np.uint8)

print("[SYSTEM] Cloak activated. Press ESC to stop.")

# =====================================================
# Main Processing Loop
# =====================================================

while True:
    grabbed, current_frame = camera.read()
    if not grabbed:
        break

    current_frame = cv2.bilateralFilter(current_frame, 9, 75, 75)

    # ----------- DeepLab Person Segmentation -----------
    input_tensor = preprocess(current_frame).unsqueeze(0)

    with torch.no_grad():
        prediction = segmentation_model(input_tensor)["out"][0]

    class_map = prediction.argmax(0).cpu().numpy()

    # COCO Person Class ID = 15
    person_region = np.where(class_map == 15, 255, 0).astype(np.uint8)
    person_region = cv2.resize(person_region, (640, 480))

    # ----------- Basic Face Preservation Logic -----------
    rows, cols = np.where(person_region == 255)
    face_protection = np.zeros_like(person_region)

    if len(rows) > 0:
        top = np.min(rows)
        bottom = np.max(rows)
        head_area = int((bottom - top) * 0.30)
        face_protection[top:top + head_area, :] = 255

    # ----------- Cloth Detection (HSV Based) -----------
    hsv_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    cloak_region = cv2.inRange(hsv_frame, blue_lower, blue_upper)

    cloak_region = cv2.bitwise_and(cloak_region, cloak_region, mask=person_region)
    cloak_region = cv2.bitwise_and(cloak_region, cv2.bitwise_not(face_protection))

    # ----------- Mask Refinement -----------
    cloak_region = cv2.morphologyEx(cloak_region, cv2.MORPH_CLOSE, morph_kernel)
    cloak_region = cv2.GaussianBlur(cloak_region, (21, 21), 0)

    edge_map = cv2.Canny(cloak_region, 50, 150)
    edge_map = cv2.GaussianBlur(edge_map, (21, 21), 0)

    refined_mask = cv2.subtract(cloak_region, edge_map)

    # ----------- Cloak Effect Composition -----------
    inverse_mask = cv2.bitwise_not(refined_mask)

    background_part = cv2.bitwise_and(static_background, static_background, mask=refined_mask)
    visible_part = cv2.bitwise_and(current_frame, current_frame, mask=inverse_mask)

    output_frame = cv2.add(background_part, visible_part)

    # ----------- Save and Display -----------
    video_output.write(output_frame)

    cv2.imshow("Original Frame", current_frame)
    cv2.imshow("Person Segmentation", person_region)
    cv2.imshow("Face Protected", face_protection)
    cv2.imshow("Cloak Mask", refined_mask)
    cv2.imshow("Final Output", output_frame)

    if cv2.waitKey(1) == 27:
        print("[SYSTEM] Shutting down...")
        break

# ----------- Cleanup -----------
camera.release()
video_output.release()
cv2.destroyAllWindows()

print("[SYSTEM] Process completed.")
