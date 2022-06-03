import os
import sys
import numpy as np
import cv2

def draw_landmarks(img, kps, color=(0, 255, 0), size=2, show_idx=False, edge_list=None, with_confidence_score=False):
    if edge_list is not None:
        for e in edge_list:
            cv2.line(img, (int(kps[e][0][0]), int(kps[e][0][1])), (int(kps[e][1][0]), int(kps[e][1][1])), color, size,
                     cv2.LINE_AA)
    # for idx in range(kps.shape[0]):
    #     x = int(kps[idx][0])
    #     y = int(kps[idx][1])
    #     if with_confidence_score:
    #         if kps[idx].shape[0] != 3:
    #             print('cannot find keypoint confidence!')
    #             import ipdb; ipdb.set_trace()
    #         pt_color = int(kps[idx][2] * 255)
    #         if pt_color < 125:
    #             continue
    #         pt_color = (256-pt_color, pt_color, pt_color)
    #     else:
    #         pt_color = color
    #     if not show_idx:
    #         cv2.circle(img, (x, y), size-1, pt_color, -1)
    #     else:
    #         cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pt_color)
    return img


def draw_hand_landmarks(img, kps, color=(255, 0, 0), size=2, show_idx=False, edges_list=None):
    if edges_list is not None:
        for idx, edges in enumerate(edges_list):
            for e in edges:
                color_lvl = 255 / 8 * (idx + 3)
                pt_color = (255, color_lvl, 1 - color_lvl)
                cv2.line(img, (int(kps[e][0][0]), int(kps[e][0][1])), (int(kps[e][1][0]), int(kps[e][1][1])), pt_color,
                         size, cv2.LINE_AA)
    # for idx in range(kps.shape[0]):
    #     x = int(kps[idx][0])
    #     y = int(kps[idx][1])
    #     if not show_idx:
    #         cv2.circle(img, (x, y), size, color, -1)
    #     else:
    #         cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return img


def draw_body_parts(img, landmarks, size=2, with_confidence_score=False):
    num_keypoints = landmarks.shape[0]
    # print('landmark = ',landmarks)
    if num_keypoints == 135:
        num_pose = 23
        num_hand = 21
        num_face = 70
        pose_edges = [[0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6]]
    elif num_keypoints == 137:
        num_pose = 25
        num_hand = 21
        num_face = 70
        pose_edges = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7]]
    elif num_keypoints == 121:
        num_pose = 9
        num_hand = 21
        num_face = 70
        pose_edges = [[1, 4], [1, 2], [2, 3], [4, 5], [5, 6]]
    elif num_keypoints == 533:
        num_pose = 23
        num_hand = 21
        num_face = 468
        pose_edges = [[11, 12], [12, 14], [11, 13], [14, 16], [13, 15]]
        # pose_edges = [[11,12], [12,14], [2,3], [3,4], [5,6], [6,7]]
    else:
        raise NotImplementedError('Unsupported number of keypoints: %d' % num_keypoints)
    pose = landmarks[:num_pose]
    face = landmarks[num_pose:num_pose + num_face]
    hand_left = landmarks[num_pose + num_face:num_pose + num_face + num_hand]
    hand_right = landmarks[num_pose + num_face + num_hand:num_pose + num_face + num_hand * 2]
    # print('dim pose = ',len(pose))
    # print('dim face = ',len(face))
    # print('dim hand_left = ',len(hand_left))
    # print('dim hand_right = ',len(hand_right))
    i = 0
    '''
    hand_edges = [
        [ [0,1], [0,5], [0,17], [5,17] ],
        [ [0,1], [0,5], [0,17], [5,17] ]
    ]
    face_edges = [
          [10,454],[454,152],[152,234],[234,10]
        ]
    '''
    hand_edges = [
        [[15, 21], [15, 19], [15, 17], [19, 17]],
        [[16, 22], [16, 20], [16, 18], [18, 20]]
    ]
    face_edges = [
        [10, 9], [8, 6], [6, 5], [5, 4], [4, 0], [2, 3], [3, 7]
    ]
    # print('face = ',face)
    # print('pose = ',pose)
    # print('hand_left',hand_left)
    # print('hand_right',hand_right)
    draw_landmarks(img, pose, color=(25, 175, 25), size=size + 2, show_idx=False, edge_list=pose_edges)
    # draw_landmarks(img, pose, color=(0,0,0), size=size, show_idx=False, edge_list=pose_edges)

    draw_landmarks(img, pose, color=(100, 100, 100), size=size, show_idx=False, edge_list=face_edges)

    draw_hand_landmarks(img, pose, color=(255, 0, 0), size=size + 1, show_idx=False, edges_list=hand_edges)

    draw_hand_landmarks(img, pose, color=(255, 0, 0), size=size + 1, show_idx=False, edges_list=hand_edges)

    return img



def formatted_print(digits):
    for i, val in enumerate(digits):
        print(val, end=', ')
        if i % 10 == 9:
            print()


def parsing_npz_137_mean_std(npz_path):
    delete_idx = list(range(24,33))


    items = np.load(npz_path, allow_pickle=True)
    mean = items['mean']
    std = items['std']

    mean = np.delete(mean, delete_idx, axis=2)
    std = np.delete(std, delete_idx, axis=2)

    print('\nmean:', mean.shape)
    formatted_print(list(mean.reshape(-1)))
    print('\nstd:', std.shape)
    formatted_print(list(std.reshape(-1)))

    print('\n')
    return mean, std


def vis_mean_pose(mean):
    W = 1280
    H = 1280 // 16 * 9

    img = np.zeros([H, W, 3], dtype=np.uint8) + 240
    mean = mean[0].transpose(1, 0)

    # head
    mean[9:39, :] += mean[39:40, :]
    mean[40:79, :] += mean[39:40, :]

    # hand
    mean[79:100, :] += mean[6:7, :]
    mean[100:121, :] += mean[3:4, :]

    mean[:, 0] += W // 2
    mean[:, 1] += H // 2

    draw_body_parts(img, mean)
    cv2.imshow('0', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #print(list(range(24,33)))
    #assert len(sys.argv) == 2
    npz_path = 'mean_std-parted.npz'
    #assert os.path.exists(npz_path)

    mean, std = parsing_npz_137_mean_std(npz_path)
    #vis_mean_pose(mean)