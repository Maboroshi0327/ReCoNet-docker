import cv2

import torch

from network import ReCoNet
from datasets import toTensor255


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    model = ReCoNet().to(device)
    model.load_state_dict(torch.load("./models/mosaic7.pth", weights_only=True))
    # model.load_state_dict(torch.load("./models/mosaic_noFTL1.pth", weights_only=True))
    # model.load_state_dict(torch.load("./models/Coco2014_epoch_2_batchSize_4.pth", weights_only=True))

    video_path = "D:\\Datasets\\video2.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape != (360, 640, 3):
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        input_tensor = toTensor255(frame).unsqueeze(0).to(device)

        # Pass the input tensor through the model
        with torch.no_grad():
            _, output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(0, 255)

        # Convert output tensor back to image format
        output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_image = output_image.astype("uint8")

        # Display the result
        cv2.imshow("Frame", output_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
