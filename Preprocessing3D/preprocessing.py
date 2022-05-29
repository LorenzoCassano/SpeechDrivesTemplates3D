import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

def rename():
    """
    Using whether the names of videos have whithespaces
    """
    folders = os.listdir('nome/')
    for file in folders:
        new_name = file.replace(' ','')
        os.rename('nome/' + file,'nome/' + new_name)



def array_trasformation(l,default = 0):
    """
    Convert a list in multidimensional array composed by:
    coordinate x, coordinate y, coordinate z and confidence
    """
    new_list = []
    i = 0
    while(i < len(l)):
        toAdd = []
        for j in range(0,3):
            toAdd.append(l[i])
            i = i +1
        if default != 0:
            toAdd.append(float(0.9))
        else:
            toAdd.append(l[i])
            i = i + 1
        new_list.append(np.array(toAdd))
        toAdd.clear()

    ou = np.array([new_list])
    return ou

def saving(pose_key,face_key,hand_left,right_hand,fold,file_name):
    """
    Save the keypoints in a npy file
    """
    if pose_key.shape == (1, 33, 4) and face_key.shape == (1, 468, 4) and hand_left.shape == (
            1, 21, 4) and right_hand.shape == (1, 21, 4):
        # only collect frames with complete pose predictions
        npy = np.concatenate([pose_key, face_key, hand_left, right_hand], axis=1).squeeze()

        npy = npy.transpose(1, 0)


        file = file_name.removesuffix('.jpg')
        print("Saving ",'./output/' + fold + '/' + file)
        np.save('./output/' + fold + '/' + file, npy)


def key_points(fold):
    j = 0
    folder = os.listdir('frames/' + fold)
    #skipping = open("./Skipping/Skipping_" + fold +'.txt',"w")
    for file in folder:
            #print("File =",file)
            pose_keypoints = []
            face_keypoints = []
            hand_left_keypoints = []
            hand_right_keypoints = []

            image = cv2.imread('frames/' + fold + '/' + file)

            # Predictions for body
            with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                              min_detection_confidence=0.5) as pose:
                image_height, image_width, _ = image.shape
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


                if not results.pose_landmarks:
                    #skipping.write(file + '\n')
                    continue
                for i in range (0,len(results.pose_landmarks.landmark)):
                    pose_keypoints.append(results.pose_landmarks.landmark[i].x)
                    pose_keypoints.append(results.pose_landmarks.landmark[i].y)
                    pose_keypoints.append(results.pose_landmarks.landmark[i].z)
                    pose_keypoints.append(results.pose_landmarks.landmark[i].visibility)

            # Predictions for hands
            with mp_hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as hands:
                resultsHands = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not resultsHands.multi_hand_landmarks:
                    #skipping.write(file + '\n')
                    continue
                c = 0
                for hand_landmarks in resultsHands.multi_hand_landmarks:
                        for i in range(0,21):
                           if c == 0:
                                hand_left_keypoints.append(hand_landmarks.landmark[i].x)
                                hand_left_keypoints.append(hand_landmarks.landmark[i].y)
                                hand_left_keypoints.append(hand_landmarks.landmark[i].z)
                           else:
                                hand_right_keypoints.append(hand_landmarks.landmark[i].x)
                                hand_right_keypoints.append(hand_landmarks.landmark[i].y)
                                hand_right_keypoints.append(hand_landmarks.landmark[i].z)
                        c = c + 1

            # Predictions for face
            with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5) as face_mesh:
                resultsFace = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not resultsFace.multi_face_landmarks:
                    #skipping.write(file + '\n')
                    continue
                for face_landmarks in resultsFace.multi_face_landmarks:
                    for i in range(0, 468):
                        face_keypoints.append(face_landmarks.landmark[i].x)
                        face_keypoints.append(face_landmarks.landmark[i].y)
                        face_keypoints.append(face_landmarks.landmark[i].z)

            try:
                pose = array_trasformation(pose_keypoints,0)
                face = array_trasformation(face_keypoints,1)
                hand_left = array_trasformation(hand_left_keypoints,1)
                hand_right = array_trasformation(hand_right_keypoints,1)
                saving(pose,face,hand_left,hand_right,fold,file)

                pose_keypoints.clear()
                hand_right_keypoints.clear()
                hand_left_keypoints.clear()
                face_keypoints.clear()
                j = j + 1
            except Exception as e:
                #skipping.write(file + '\n')
                #print("Skipping ",e)
                pass
    #skipping.close()
    print("Saved ",j,"/",len(folder))

def main():
    folders = os.listdir('frames/')
    for folder in folders:
        dest = 'output/' + folder
        print("Starting folder ",folder)

        # create folders if needed
        if not os.path.isdir(dest):
            os.mkdir(dest)
            print("\tCreated dir " + dest)
        key_points(folder)



if __name__ == '__main__':
    main()