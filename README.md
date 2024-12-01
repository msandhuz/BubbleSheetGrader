# Bubble Sheet Grader

A Python-based tool that automatically grades bubble sheet style exams. The script processes scanned images of bubble sheets, identifies the bubbles corresponding to answers, and calculates the score based on a predefined answer key.

## Features

- Detects bubble sheet from an image.
- Automatically identifies the correct answers based on a predefined answer key.
- Grades the exam by comparing bubbled answers to the correct answers.
- Outputs the results in percentage format.
- Saves the processed images with the results in a `results` folder.

## Prerequisites

To run this project, you need to install the following Python libraries:

- OpenCV
- NumPy
- imutils
- argparse

You can install the dependencies with:

```bash
pip install -r requirements.txt
