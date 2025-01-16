import cv2
from face_processing.face_matcher_models.face_matcher import FaceMatcherModels


class Process:
    def __init__(self):
        self.matcher = FaceMatcherModels()

    def main(self):
        face_1 = cv2.imread('process/face_1.jpeg')
        face_2 = cv2.imread('process/face_2.jpg')

        matching, distance = self.matcher.face_matching_vgg_model(face_1, face_2)
        print(f'Matcher: {matching} similarity: {distance}')


matcher = Process()
matcher.main()
