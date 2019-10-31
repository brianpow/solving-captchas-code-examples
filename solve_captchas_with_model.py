from keras.models import load_model
import helpers
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os.path

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

# loop over the image paths
for image_file in captcha_image_files:
    # Load the image
    image = cv2.imread(image_file)

    gray, thresh = helpers.pre_processing(image)
    
    _, letter_image_regions = helpers.find_contours(thresh)

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    filename = os.path.basename(image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    if len(letter_image_regions) != len(captcha_correct_text):
        print("[ERROR] Finding contours from {} failed, expected {}, found {}".format(image_file, len(captcha_correct_text),str(len(letter_image_regions))))
        continue

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create a list to hold output images and our predicted letters
    predictions = []
    outputs = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        output = cv2.merge([gray] * 3)
    
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = helpers.resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        outputs.append(output)
    # Print the captcha's text
    captcha_text = "".join(predictions)
    if captcha_text == captcha_correct_text:
        print("[SUCCESS] Solving {} successful! Answer: {}".format(image_file, captcha_text))
    else:
        print("[FAILED] Solving {} failed! Guessed: {}, answer: {}".format(image_file, captcha_text, captcha_correct_text))

    # Show the annotated image
    cv2.imshow("Output", np.concatenate(outputs,axis=0))
    cv2.waitKey()