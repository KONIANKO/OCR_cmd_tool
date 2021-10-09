# OCR_cmd_tool
 
 A tool to extract text from images and pdf files.

## Accepted input file formats: 
 png, jpg, pdf

## Usage:


 1. Put input files in into the same directory, as ocr_tool.py
 2. In command line type:

 	python ocr_tool.py --input=./input_file_name.png --output=output_file_name.txt

 3. --verbose is not supported by python > 3.6

## Requirements:
 1. install requiered packages by executing following command in command line:

    pip install -r requirements.txt

 2. You will have to install tesseract-ocr-3.02.02 and add it to PATH
