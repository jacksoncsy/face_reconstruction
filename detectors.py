import numpy as np


class iBugDetectors(object):
    def __init__(self, device="cuda:0"):
        from ibug.face_detection import RetinaFacePredictor
        from ibug.face_alignment import FANPredictor

        # Create a RetinaFace detector using Resnet50 backbone
        self.face_detector = RetinaFacePredictor(
            threshold=0.8,
            device=device,
            model=RetinaFacePredictor.get_model('resnet50'),
        )

        # Create a facial landmark detector
        self.landmark_detector = FANPredictor(
            device=device,
            model=FANPredictor.get_model('2dfan2_alt'),
        )

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        # Detect faces from the image
        # Result is a Nx15 (for RetinaFace) matrix, in which N is the number of detected faces.
        # The first 4 columns store the left, top, right, and bottom coordinates of the
        # detected face boxes. The 5th columns stores the detection confidences.
        # The remaining columns store the coordinates (in the order of x1, y1, x2,
        # y2, ...) of the detected landmarks.
        detected_faces = self.face_detector(image, rgb=True)

        if len(detected_faces) == 0:
            return [0], 'kpt68'

        # select only the top-1 rank face
        detection_scores = detected_faces[:, 4]
        idx = np.argmax(detection_scores)
        detected_faces = detected_faces[[idx]]

        # Detect landmarks from the faces
        kpt, _ = self.landmark_detector(image, detected_faces, rgb=True)
        kpt = kpt[0]

        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        bbox = [left, top, right, bottom]
        return bbox, 'kpt68'


class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0])
            right = np.max(kpt[:,0])
            top = np.min(kpt[:,1])
            bottom = np.max(kpt[:,1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'


class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'



