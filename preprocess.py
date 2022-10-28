import cv2

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# v1 20221028 Qin Xiong
def extract_faces_from_video(path):
    video = cv2.VideoCapture(path)
    totalframe = video.get(cv2.CAP_PROP_FRAME_COUNT)
    interval = totalframe / 1000
    for i in range(0, 1000):
        fid = i * interval
        video.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video.read()
        # look for face, only 1
        face = extract_faces_from_image(frame)[0]
        # save
        (x, y, w, h) = face
        cv2.imwrite(f"{path}-{fid}.jpg", frame[y:y + h, x:x + w])


def extract_faces_from_image(image):
    faces = classifier.detectMultiScale(image, 1.3, 4)
    return faces
