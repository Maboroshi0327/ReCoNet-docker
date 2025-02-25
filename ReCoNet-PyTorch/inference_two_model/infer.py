import sys
sys.path.append("/root/ReCoNet")

import cv2
import torch
from network import ReCoNet
from utilities import Inference


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_1_infer = Inference(ReCoNet, 1, "./models/starry-night-1.pth", "../datasets/video2.mp4", device)
    model_2_infer = Inference(ReCoNet, 1, "./models/starry-night-2.pth", "../datasets/video2.mp4", device)

    for output_image_1, output_image_2 in zip(model_1_infer, model_2_infer):
        cv2.imshow("starry-night-1", output_image_1)
        cv2.imshow("starry-night-2", output_image_2)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
