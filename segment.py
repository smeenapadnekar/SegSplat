import cv2
import numpy as np
import matplotlib.pyplot as plt 
 
bg_color = [255, 255, 255]  # White background
# image_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/Tanks/Barn/test/ours_10000/gt/000521.png"
# seg_file ="/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/Tanks/Barn/test/ours_10000/test_seg/000521.png"

# image_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/counter/test/ours_10000/gt/DSCF5857.png"
# seg_file ="/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/counter/test/ours_10000/test_seg/DSCF5857.png"
image_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/fern/test/ours_10000/gt/IMG_4026.png"
seg_file = "/media/vplab/6b43b65c-0636-45e8-b1a2-033acce48735/Meena/3DGD/FewShot/joint_gs_seg/output/fern/test/ours_10000/test_seg/IMG_4026.png"
img = cv2.imread(image_file)
seg = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)

for i in range(200):
    mask = (seg == i).astype(np.uint8)
    if np.sum(mask) == 0:
        continue  # Skip empty masks

    # Find bounding box of the mask
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Crop the mask and original image
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    # Create 3-channel mask
    mask_3c = np.stack([cropped_mask]*3, axis=-1)

    # Create white background
    background = np.full_like(cropped_img, fill_value=255)

    # Composite object on white background
    object_on_white = np.where(mask_3c == 1, cropped_img, background)

    # Show result
    plt.figure()
    plt.imshow(cv2.cvtColor(object_on_white, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
