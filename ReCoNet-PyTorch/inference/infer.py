import sys
sys.path.append("/root/ReCoNet")

import cv2
import torch
from network import ReCoNet, ReCoNetSD1, ReCoNetSD2
from utilities import Inference


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_infer = Inference(ReCoNet, 1, "./models/starry-night-2.pth", "../datasets/video3.mp4", device)
    # model_infer = Inference(ReCoNetSD1, 1, "./models_old/SD1_epoch_4_batchSize_2.pth", "../datasets/video3.mp4", device)
    # model_infer = Inference(ReCoNetSD2, 1, "./models_old/SD2_epoch_4_batchSize_2.pth", "../datasets/video3.mp4", device)
    # model_infer = Inference(ReCoNet, 1, "./models/Coco2014_epoch_10_batchSize_4.pth", "../datasets/video3.mp4", device)

    for output_image in model_infer:
        cv2.imshow("Frames", output_image)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
