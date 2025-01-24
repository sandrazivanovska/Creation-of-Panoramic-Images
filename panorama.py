import cv2
import argparse

parser = argparse.ArgumentParser(description='Stitching images')
parser.add_argument('image_files', nargs='+', help='paths to images')
args = parser.parse_args()

image_list = []
for file_path in args.image_files:
    img = cv2.imread(file_path)
    if img is None:
        print(f"Failed to load image: {file_path}")
        exit(1)
    image_list.append(img)

target_width = 400
scaled_images = []
for image in image_list:
    scaling_ratio = target_width / image.shape[1]
    target_height = int(image.shape[0] * scaling_ratio)
    scaled_img = cv2.resize(image, (target_width, target_height))
    scaled_images.append(scaled_img)

gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in scaled_images]

sift_detector = cv2.SIFT_create()
keypoints_list = []
descriptors_list = []
for gray_image in gray_images:
    kp, desc = sift_detector.detectAndCompute(gray_image, None)
    keypoints_list.append(kp)
    descriptors_list.append(desc)

bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
descriptor_matches = []
for i in range(len(descriptors_list) - 1):
    descriptor_matches.append(bf_matcher.match(descriptors_list[i], descriptors_list[i + 1]))

sorted_descriptor_matches = [sorted(match, key=lambda x: x.distance) for match in descriptor_matches]

top_N_matches = 50
matches_images = []
for i in range(len(sorted_descriptor_matches)):
    match_img = cv2.drawMatches(
        scaled_images[i], keypoints_list[i], scaled_images[i + 1], keypoints_list[i + 1],
        sorted_descriptor_matches[i][:top_N_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_images.append(match_img)

stitcher_instance = cv2.Stitcher.create()
stitch_status, panorama = stitcher_instance.stitch(scaled_images)

if stitch_status == cv2.Stitcher_OK:
    cv2.imshow('Panorama Result', panorama)
else:
    print(f"Stitching failed with status {stitch_status}")

for idx, match_img in enumerate(matches_images):
    cv2.imshow(f'Keypoint Matches {idx + 1}-{idx + 2}', match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
