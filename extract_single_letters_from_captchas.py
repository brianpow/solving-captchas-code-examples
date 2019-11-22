import os
import os.path
import cv2
import glob
import imutils
import helpers
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Extract letters from captcha image.')
parser.add_argument('project', nargs='?', default="default",
                    help='name of the project (subfolders of the required image files)')
parser.add_argument('-n','--number', type=int, default=999999,
                    help='Number of images to be processed')
parser.add_argument('-f','--failed', action='store_true', default=False,
                    help='Preview image failed to find contours.')
parser.add_argument('-s','--success', action='store_true', default=False,
                    help='Preview image with correct contours found.')
args = parser.parse_args()

CAPTCHA_IMAGE_FOLDER = os.path.join(args.project, "generated_captcha_images")
OUTPUT_FOLDER = os.path.join(args.project, "extracted_letter_images")


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

if len(captcha_image_files) == 0:
    print("[ERROR] No image found at {}".format(CAPTCHA_IMAGE_FOLDER))
    exit(1)
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    if args.number == i:
        print("[INFO] {} files limited reached. See help to override".format(args.number) )
        break

    print("[INFO] processing {} ({}/{})".format(captcha_image_file, i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image
    image = cv2.imread(captcha_image_file)
    
    gray, thresh = helpers.pre_processing(image)

    detected_contours, filtered_contours = helpers.find_contours(thresh)

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    outputs = []

    if len(filtered_contours) != len(captcha_correct_text):
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    # loop over the lektters
    for letter_bounding_box in filtered_contours:
        output = cv2.merge([thresh] * 3)

        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # draw the prediction on the output image
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

        outputs.append(output)

    # Show the annotated image
    if len(filtered_contours) != len(captcha_correct_text):
        print("[ERROR] Finding contours from {} failed, expected: {}, detected: {}, filtered: {}".format(captcha_image_file, len(captcha_correct_text),str(len(detected_contours)),str(len(filtered_contours))))
        if args.failed:
            cv2.imshow(filename, np.concatenate(outputs,axis=0))
            cv2.waitKey()
        continue
    elif args.success:
        cv2.imshow(filename, np.concatenate(outputs,axis=0))
        cv2.waitKey()


    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    filtered_contours = sorted(filtered_contours, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(filtered_contours, captcha_correct_text):
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
