import os

import cv2
import numpy as np
from retinaface import RetinaFace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = RetinaFace.build_model()


def detect_and_save_faces(input_image_path, output_folder, face_idx):
    # Load the input image
    image = cv2.imread(input_image_path)

    try:
        # Detect faces using the SSD model
        detected_faces = RetinaFace.extract_faces(
            input_image_path,
            threshold=0.97,
            model=model,
            align=True,
            align_first=True,
            allow_upscaling=True,
            expand_face_area=10,
        )

        if len(detected_faces) > 1:
            # print(f'{input_image_path} has more than 1 face.')
            return face_idx

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Loop through each detected face and save it as a 300x300 PNG file
        for i, face_info in enumerate(detected_faces):
            face = face_info
            # print(face)

            # print("resizing..")
            maxwidth, maxheight = 300, 300
            f1 = maxwidth / face.shape[1]
            f2 = maxheight / face.shape[0]
            f = min(f1, f2)
            dim = (int(face.shape[1] * f), int(face.shape[0] * f))
            face_resized = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)

            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Create a black image with the target size
            padded_image = np.zeros((300, 300, 3), dtype=np.uint8)
            padding_rows = (300 - dim[1]) // 2
            padding_cols = (300 - dim[0]) // 2

            # Add the resized image to the padded image, with padding on the left and right sides
            padded_image[padding_rows : padding_rows + dim[1], padding_cols : padding_cols + dim[0]] = face_rgb

            face_idx += 1
            # Save the face as a PNG file in the output folder
            output_path = os.path.join(output_folder, f"face_{face_idx}.png")
            cv2.imwrite(output_path, padded_image)

        # print(f"{len(detected_faces)} faces detected and saved to {output_folder}")
    except Exception as e:
        print(str(e))

    return face_idx


import os

path_img = "images"
path_faces = "faces"

# # test
# input_image_path = '01.png'
# output_folder = 'output/'
# detect_and_save_faces(input_image_path, output_folder, 0)

for gender in sorted(os.listdir(path_img)):
    path_gender = os.path.join(path_img, gender)
    for name in sorted(os.listdir(path_gender)):
        if os.path.exists(os.path.join(path_faces, gender, name)):
            continue

        path_name = os.path.join(path_img, gender, name)
        face_idx = 0
        for path_file in sorted(os.listdir(path_name)):
            path_img_file = os.path.join(path_name, path_file)
            if os.path.getsize(path_img_file) < 100000:
                continue
            # print(path_img_file)
            # input_image_path = '01.png'
            output_folder = os.path.join(path_faces, gender, name)
            face_idx = detect_and_save_faces(path_img_file, output_folder, face_idx)
