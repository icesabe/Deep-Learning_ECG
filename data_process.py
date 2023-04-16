import cv2
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm 

# import matplotlib.pyplot as plt


def process_single_fig(file, outfile):
    ECG = cv2.imread(file)
    img_gray = cv2.cvtColor(ECG, cv2.COLOR_BGR2GRAY)

    # GET MASK
    h, w = img_gray.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # OTSU threshold the image
    _, th_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    # threshold
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # move horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(ECG, [c], -1, (255, 255, 255), 1)
    # move vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(ECG, [c], -1, (255, 255, 255), 1)
    gray = cv2.cvtColor(ECG, cv2.COLOR_BGR2GRAY)
    # Column Count
    row_count = np.zeros((gray.shape[0]))
    col_count = np.zeros((gray.shape[1]))
    for i, row in enumerate(gray):
        # row_count[i] +=
        for j, col in enumerate(row):
            if col < 125:
                col_count[j] += 1
                row_count[i] += 1

    for i, row in enumerate(gray):
        for j, col in enumerate(row):
            if 50 < col_count[j]:
                # img_gray[i][j] = 255
                if 100 < row_count[i] and gray[i][j] < 50:
                    gray[i][j] = 255
    # final result with mask
    res = cv2.bitwise_and(gray, gray, mask=mask)

    # save img
    cv2.imwrite(outfile, res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train entrance')
    # ===== Data related parameters =================
    parser.add_argument('--path', type=str, default='/Users/chain/git/AI-medical/ECG')
    parser.add_argument('--in_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='process_1')
    args = parser.parse_args()
    all_files = [] 
    for filename in os.listdir(args.in_dir):
        if 'png' in filename and not os.path.isfile(os.path.join(args.out_dir, filename)):
            all_files.append(filename)
    print(f"{len(all_files)} files to be processed. ")
    ignore_num = 0
    for i in tqdm(range(len(all_files))):
        filename = all_files[i]
        try:
            process_single_fig(os.path.join(args.in_dir, filename),
                            os.path.join(args.out_dir, filename))
        except: 
            print("ERROR file: ")
            print(filename)
            ignore_num += 1
            # sys.exit()
    print(f"there are {ignore_num} file in the wrong format that can not be processed.")
    
