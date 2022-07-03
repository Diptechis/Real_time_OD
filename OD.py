import cv2

# Open CV DNN

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Class List

classes = []

with open('dnn_model/classes.txt', 'r') as file_obj:
    for class_name in file_obj.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize Camera

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



while True:
    # Get Frame
    ret, frame = cap.read()

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(150, 35, 70), 2 )
        cv2.rectangle(frame, (x, y), (x+w, y+h), (150, 35, 70), 3)



    # print("Class IDS: ", class_ids)
    # print("Score: " , scores)
    # print("Bboxes:", bboxes)


    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


