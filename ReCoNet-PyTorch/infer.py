import cv2

import torch

from network import ReCoNet
from datasets import toTensor255


device = "cuda" if torch.cuda.is_available() else "cpu"
input_frame_num = 4


def frame_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape != (360, 640, 3):
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    return toTensor255(frame)


if __name__ == "__main__":
    model = ReCoNet(input_frame_num).to(device)
    model.load_state_dict(torch.load("./models/mosaic66.pth", weights_only=True))
    # model.load_state_dict(torch.load("./models/mosaic_noFTL1.pth", weights_only=True))
    # model.load_state_dict(torch.load("./models/Coco2014_epoch_2_batchSize_4.pth", weights_only=True))

    video_path = "../datasets/video2.mp4"
    cap = cv2.VideoCapture(video_path)

    # Read the first few frames
    imgs = list()
    for i in range(input_frame_num):
        ret, frame = cap.read()
        imgs.append(frame_to_tensor(frame))

    # Start the video style transfer
    while True:
        # Pass the input tensor through the model
        with torch.no_grad():
            input_tensor = torch.cat(imgs, dim=0).unsqueeze(0).to(device)
            _, output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(0, 255)

        # Convert output tensor back to image format
        output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_image = output_image.astype("uint8")

        # Display the result
        cv2.imshow("Frame", output_image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update the input tensor list
        imgs.pop(0)
        imgs.append(frame_to_tensor(frame))

    cap.release()
    cv2.destroyAllWindows()
