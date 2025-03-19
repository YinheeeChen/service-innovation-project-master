# import os
# from PIL import Image
# import numpy as np
#
# count = 0
#
# def save_image(image, folder_path):
#     global count
#     count += 1
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     image = Image.fromarray(np.uint8(image))
#     image.save(os.path.join(folder_path, str(count) + ".png"))
#
# def get_image(image):
#     return Image.fromarray(np.uint8(image))
# # import cv2
# # from PIL import Image
# # import os
# #
# #
# # IMG_DIR = 'IMG'
# # if not os.path.exists(IMG_DIR):
# #     os.makedirs(IMG_DIR)
# # def save_frames_as_images(video_path, output_folder):
# #
# #     cap = cv2.VideoCapture(video_path)
# #     frame_count = 0
# #
# #     while(cap.isOpened()):
# #
# #         ret, frame = cap.read()
# #         if ret:
# #
# #             Image.fromarray(frame).save(os.path.join(output_folder, f'frame_{frame_count}.png'))
# #             frame_count += 1
# #         else:
# #             break
# #
# #
# #     cap.release()
# #
# #
# # save_frames_as_images('videos/episode-8166-reward-403.0.npy.mp4', IMG_DIR)
import numpy as np
from PIL import Image
import os
from fp import preprocess
IMG_DIR = 'imgs'
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

npy_file = 'episode-8166-reward-403.0.npy'

# Print the .npy file path
print(f'Loading .npy file from: {npy_file}')

# Load the .npy file
frames = np.load(npy_file)
preprocessed_frames = preprocess(frames)

# Print the shape and data type of the frames
print(f'Frames shape: {frames.shape}, data type: {frames.dtype}')

# Iterate over the frames
for i in range(preprocessed_frames.shape[0]):
    # Here, frames[i] is the current state of the game
    # You can process or save the state as needed
    # For example, you can save the state as an image
    img_path = os.path.join(IMG_DIR, f'frame_{i}.png')
    Image.fromarray(preprocessed_frames[i]).save(img_path)

    # Print the image path
    print(f'Saved image to: {img_path}')