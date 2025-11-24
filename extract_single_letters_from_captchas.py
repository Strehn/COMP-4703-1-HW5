"""
Homework Part 1: Letter Extraction from CAPTCHAs

Goal: Implement robust preprocessing and segmentation to extract single-letter images
from generated CAPTCHA images. Save each letter into a folder named by its class label.

Tasks:
- TODO: Parameterize input/output via CLI arguments (already scaffolded below with argparse).
- TODO: Handle edge cases (no contours, wrong number of letters) and report counts.

Expected Output:
- A directory tree under OUTPUT_FOLDER where each subfolder is a label and contains
  extracted letter images.
"""

import os
import os.path
import cv2
import glob
import imutils
import argparse
from typing import Tuple


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"

# Argument parsing for assignment reproducibility
parser = argparse.ArgumentParser(description="Extract single letters from CAPTCHA images")
parser.add_argument("--input-folder", default=CAPTCHA_IMAGE_FOLDER, help="Folder containing CAPTCHA images")
parser.add_argument("--output-folder", default=OUTPUT_FOLDER, help="Folder to write extracted letter images")
parser.add_argument("--max-images", type=int, default=None, help="Optional: limit number of images processed")
# TODO (optional): Add a flag like --morph-open to enable morphological opening.
args = parser.parse_args()

CAPTCHA_IMAGE_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))[: args.max_images] if args.max_images else glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

# Reporting counters
skipped_bad_letters = 0
skipped_no_contours = 0
skipped_unreadable = 0
processed = 0

counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    processed += 1

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    if image is None:
        skipped_unreadable += 1
        print(f"[WARN] Could not read image: {captcha_image_file}")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # TODO: Consider experimenting with different thresholding or adaptive methods.
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # TODO (optional): enable morphological opening to remove small noise
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # find the contours (continuous blobs of pixels) in the image
    contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4.x returns (contours, hierarchy); OpenCV 3.x returns (image, contours, hierarchy)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        skipped_no_contours += 1
        print(f"[WARN] No contours found in {captcha_image_file}")
        continue

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Skip empty or invalid contours
        if contour is None or len(contour) == 0:
            continue

        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        if h == 0:
            continue

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        skipped_bad_letters += 1
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1

# Summary report for the assignment write-up
print("\n[REPORT] Extraction summary:")
print(f"  Processed images: {processed}")
print(f"  Skipped unreadable: {skipped_unreadable}")
print(f"  Skipped no contours: {skipped_no_contours}")
print(f"  Skipped wrong letter count: {skipped_bad_letters}")
