import click
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from scipy.ndimage import interpolation as inter


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


#erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image,
                                 M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated


# accept input in command line with --
# --verbose not supported by python > 3.6
@click.command(context_settings=dict(ignore_unknown_options=True, ))
@click.option('--input', prompt=True, required=True)
@click.option('--output', prompt=True, required=True)
def ocr_tool(input, output):

    input_file_name = input
    output_file_name = output

    # handle pdf input file
    if input_file_name.endswith(('.pdf')):

        # extract number of pages in pdf
        images = convert_from_path(input_file_name, 500)

        # each page is a single image. Stack images to one long image.
        curr_image = images[0]
        for i in range(len(images) - 1):
            curr_image = np.concatenate((curr_image, images[i + 1]), axis=0)

        img = curr_image
        img = cv2.resize(img,
                         None,
                         fx=1.2,
                         fy=1.2,
                         interpolation=cv2.INTER_CUBIC)

    else:
        img = cv2.imread(input_file_name)

        # reshape image to get better results
        img = cv2.resize(img,
                         None,
                         fx=3.0,
                         fy=3.0,
                         interpolation=cv2.INTER_CUBIC)

    # convert to grayscale
    img = get_grayscale(img)

    # noise removal
    img = cv2.fastNlMeansDenoising(img, img, 30.0, 7, 21)

    # skew correction
    #img = deskew(img)

    # perform tresholding
    img = thresholding(img)

    # print(pytesseract.image_to_string(img))

    # store exctracted text
    doc_text = pytesseract.image_to_string(img)

    # write extracted text to text file
    with open(output_file_name, "w", encoding='utf-8') as text_file:
        text_file.write(doc_text)


if __name__ == '__main__':
    ocr_tool()
