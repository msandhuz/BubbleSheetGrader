# Import necessary libraries for image processing and contour analysis
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os

# Argument parsing for input image and output file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")  # Input image path
ap.add_argument("-o", "--output", required=True,
                help="path to the output results file")  # Output file path for results
args = vars(ap.parse_args())

# Answer key for the bubble sheet, mapping question numbers to correct answers
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# Function to process the image and calculate the score
def process_image(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detection
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours in the edge-detected image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # Check if contours were found
    if len(cnts) > 0:
        # Sort contours by area (largest first)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Look for the largest quadrilateral contour (likely the paper)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:  # Found a quadrilateral (paper)
                docCnt = approx
                break

    # Perform a perspective transform to get a top-down view of the document
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # Apply Otsu's thresholding to binarize the warped image
    thresh = cv2.threshold(warped, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image (bubbles)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Loop through contours to find potential question bubbles
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # Consider only bubbles that are square-like and large enough
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    # Sort the question contours top to bottom
    questionCnts = contours.sort_contours(questionCnts,
                                          method="top-to-bottom")[0]
    correct = 0

    # Process each question's bubbles (5 possible answers per question)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        # Sort the current question's answer bubbles (left to right)
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        # Find the "bubbled" answer by comparing non-zero pixels in each bubble
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            # Identify the bubbled answer (the one with the most non-zero pixels)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # Check if the bubbled answer is correct
        color = (0, 0, 255)  # Red for incorrect
        k = ANSWER_KEY[q]
        if k == bubbled[1]:
            color = (0, 255, 0)  # Green for correct
            correct += 1

        # Draw the outline of the selected answer on the paper
        cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    # Calculate the score as percentage of correct answers
    score = (correct / 5.0) * 100
    return score, paper

# Function to process multiple images and store the results
def process_multiple_images(image_folder, output_file, results_folder):
    # Create a folder for results if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Open the output file in write mode
    with open(output_file, "w") as file:
        # Process images (test_01.png to test_05.png)
        for i in range(1, 6):
            image_path = os.path.join(image_folder, f"test_0{i}.png")
            print(f"Processing {image_path}...")

            # Process the image and calculate score
            score, paper = process_image(image_path)

            # Write the result to the file
            file.write(f"Results for {image_path}:\n")
            file.write(f"Score: {score:.2f}%\n")
            file.write("=" * 40 + "\n")

            # Save the processed image to the results folder
            output_image_path = os.path.join(results_folder, f"processed_test_0{i}.png")
            cv2.imwrite(output_image_path, paper)
            print(f"Processed image saved as {output_image_path}")

# Paths to the images, output file, and results folder
image_folder = "/Users/zainabmushtaq/Downloads/Bubble Sheet/images/"
output_file = "/Users/zainabmushtaq/Downloads/Bubble Sheet/results.txt"
results_folder = "/Users/zainabmushtaq/Downloads/Bubble Sheet/results/"

# Process images and save the results
process_multiple_images(image_folder, output_file, results_folder)

print(f"Results have been saved to {output_file}.")

# Command to run the script
# python3 test_grader.py --image images/test_01.png --output /path/to/results.txt
