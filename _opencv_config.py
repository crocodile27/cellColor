import os
# Set before cv2 loads
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

print("MAX PIXELS =", os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"])
