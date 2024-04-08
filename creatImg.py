import cv2
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import uuid 

# Directory where the captured images and annotations will be saved
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Gesture label and image count initialization
gesture_label = ""
image_count = 0

# Initialize the list to store the points for the bounding box
bbox_points = []
drawing = False  # True if mouse is pressed down
current_frame = None

# Fixed size for resizing imagesq
fixed_size = (640, 640)

def create_pascal_voc_xml(filename, folder, label, bbox, img_size):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = os.path.join(folder, filename)

    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(img_size[0])
    SubElement(size, 'height').text = str(img_size[1])
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    obj = SubElement(root, 'object')
    SubElement(obj, 'name').text = label
    SubElement(obj, 'pose').text = 'Unspecified'
    SubElement(obj, 'truncated').text = '0'
    SubElement(obj, 'difficult').text = '0'

    bndbox = SubElement(obj, 'bndbox')
    SubElement(bndbox, 'xmin').text = str(bbox[0][0])
    SubElement(bndbox, 'ymin').text = str(bbox[0][1])
    SubElement(bndbox, 'xmax').text = str(bbox[1][0])
    SubElement(bndbox, 'ymax').text = str(bbox[1][1])

    xml_str = tostring(root)
    pretty_xml_as_string = parseString(xml_str).toprettyxml()
    return pretty_xml_as_string

def save_image_and_annotation(frame, label, count, bbox):
    unique_id = f"{label}_{uuid.uuid4()}"  # Generate unique ID
    # Resize frame to fixed size
    resized_frame = cv2.resize(frame, fixed_size)

    # Save the image
    img_name = f"{label}_{unique_id}.jpg"
    cv2.imwrite(os.path.join(save_dir, img_name), resized_frame)

    # Create and save Pascal VOC XML annotation
    xml_content = create_pascal_voc_xml(img_name, save_dir, label, bbox, fixed_size)
    with open(os.path.join(save_dir, f"{label}_{unique_id}.xml"), "w") as f:
        f.write(xml_content)

    print(f"Saved {img_name} and annotation XML")

def draw_rectangle(event, x, y, flags, param):
    global bbox_points, drawing, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            temp_frame = current_frame.copy()
            cv2.rectangle(temp_frame, bbox_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox_points.append((x, y))
        cv2.rectangle(current_frame, bbox_points[0], bbox_points[1], (0, 255, 0), 2)
        cv2.imshow("Frame", current_frame)

# Initialize camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

while True:
    ret, frame = cap.read()
    #print(frame)
    if not ret:
        break
    frame = cv2.resize(frame, fixed_size) 
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        gesture_label = ["thumbs_up", "thumbs_down", "fist", "index"][int(chr(key)) - 1]
        current_frame = frame.copy()
        cv2.imshow("Frame", current_frame)
        bbox_points = []  # Reset bounding box points

        # Wait for bounding box to be drawn and confirmed
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord('y'):
                if len(bbox_points) == 2:
                    save_image_and_annotation(frame.copy(), gesture_label, image_count, bbox_points)
                    image_count += 1
                break
            elif key2 == ord('n'):
                bbox_points = []  # Reset bounding box points
                current_frame = frame.copy()
                cv2.imshow("Frame", current_frame)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
