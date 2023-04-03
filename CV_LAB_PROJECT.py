
def get_path_list(train_path):
    train_names = os.listdir(train_path)
    return train_names

def get_train_images_and_classes_id(train_path, train_names):
    image_list = []
    image_classes_list = []

    for index, name in enumerate(train_names):
        image_dir_path = train_path + '/' + name
        for image_path in os.listdir(image_dir_path):
            image_full_path = image_dir_path + '/' + image_path
            image = cv2.imread(image_full_path)
            image_list.append(image)
            image_classes_list.append(index)
    return image_list , image_classes_list





def detect_train_faces_and_filter(image_list, image_classes_list):
    train_face_grays=[]
    image_classes_list_new=[]
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for i, image in enumerate(image_list):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect_faces = face_classifier.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors = 5)
        if(len(detect_faces) > 0):
            for face in detect_faces:
                x, y, w, h = face
                face_image = image_gray[y:y+h, x:x+w]
                train_face_grays.append(face_image)
                image_classes_list_new.append(image_classes_list[i])

    return train_face_grays, image_classes_list_new


def train(train_face_grays, image_classes_list):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    return recognizer

def get_test_images_data(test_path):
    image_test_list = []
    for name in os.listdir(test_path):
        path = test_path + '/' + name
        image = cv2.imread(path)
        image_test_list.append(image)
    return image_test_list

def detect_test_faces_and_filter(image_list):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    test_faces_gray=[]
    test_faces_rects=[]
    for image in image_list:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detect_faces = face_classifier.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors = 5)
        if(len(detect_faces) > 0):
            for face in detect_faces:
                x, y, w, h = face
                face_image = image_gray[y:y+h, x:x+w]
                test_faces_gray.append(face_image)
                test_faces_rects.append(face)
    
    return test_faces_gray, test_faces_rects


    
def predict(recognizer, test_faces_gray):
    predict_results = []
    for image in test_faces_gray:
        result = recognizer.predict(image)
        predict_results.append(result)
    return predict_results

def get_agent_name(person_name):
    agent_name = 'empty'
    if(person_name == 0):
        agent_name = 'Captain America'
    if(person_name == 1):
        agent_name = 'Thor'
    if(person_name == 2):
        agent_name = 'Clint Barton'
    if(person_name == 3):
        agent_name = 'Bruce Banner'
    if(person_name == 4):
        agent_name = 'Iron Man'
    if(person_name == 5):
        agent_name = 'Nick Fury'
    if(person_name == 6):
        agent_name = 'Black Widow'
    if(person_name == 7):
        agent_name = 'Loki'
    return agent_name
        

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, width, height):
    agent_name = ''
    image_list_drawn = []
    
    for i, image in enumerate(test_image_list):

        dim = (width, height)
        image = cv2.resize(image, dim)
        person_name, _ = predict_results[i]
        agent_name = get_agent_name(person_name)

        if(person_name == 0):
            person_name = 'Chris Evans'
        if(person_name == 1):
            person_name = 'Chris Hemsworth'
        if(person_name == 2):
            person_name = 'Jeremy Renner'
        if(person_name == 3):
            person_name = 'Mark Ruffalo'
        if(person_name == 4):
            person_name = 'Robert Downey Jr'
        if(person_name == 5):
            person_name = 'Samuel L Jackson'
        if(person_name == 6):
            person_name = 'Scarlett Johansson'
        if(person_name == 7):
            person_name = 'Tom Hiddleston'
        
        x, y, w, h = test_faces_rects[i]
        cv2.rectangle(image, (x-27,y-27), (x+w-70, y+h-70), (0, 255, 0), 4)
        result_text = agent_name + ' Confirmed'
        cv2.putText(image, result_text, (x-65, y-30),cv2.FONT_HERSHEY_PLAIN, 0.9, (0,255,0), 2)
        name = str(person_name)
        cv2.putText(image, name, (x-30, y+150),cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,255), 2)
        image_list_drawn.append(image)

    return image_list_drawn

    

def combine_and_show_result(image_list):
    res = np.concatenate((image_list[0], image_list[1], image_list[2], image_list[3], image_list[4]), axis =1)
    cv2.imshow("RESULT", res)
    cv2.waitKey(0)
    return res



if __name__ == "__main__":
    import cv2
    import numpy as np
    import os

    train_path = "Dataset/Train"

    train_names = get_path_list(train_path)
    train_image_list, image_classes_list = get_train_images_and_classes_id(train_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    test_path = "Dataset/Test"
 
    test_image_list = get_test_images_data(test_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 250, 300)
    
    combine_and_show_result(predicted_test_image_list)