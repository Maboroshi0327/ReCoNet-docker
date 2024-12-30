import cv2

import torch

from network import ReCoNet
from datasets import toTensor255


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model = ReCoNet().to(device)
    model.load_state_dict(torch.load("./models/Coco2014_epoch_2_batchSize_4.pth", weights_only=True))

    video_path = "D:\\Datasets\\video2.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor and feed into the model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = toTensor255(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            _, output_tensor = model(input_tensor)

        # Convert output tensor back to image format
        output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_image = output_image.astype("uint8")

        # Display the result
        cv2.imshow("Frame", output_image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
