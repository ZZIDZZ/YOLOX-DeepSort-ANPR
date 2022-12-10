import cv2
import numpy as np
import imutils
import pytesseract

img = cv2.imread('data/2.jpg')
img = imutils.resize(img, width=500 )
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find Canny edges
edged = cv2.Canny(img_gray, 5, 20)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = len, reverse=True)[:30]
screenCnt = []
for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
                screenCnt.append(approx)
print(contours)
  
cv2.imshow('Canny Edges After Contouring', edged)
  
print("Number of Contours found = " + str(len(contours)))
# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(img, screenCnt, -1, (0, 255, 0), 3)
Cropped_loc = './7.png'
cv2.imshow("cropped", cv2.imread(Cropped_loc))
plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
print("Number plate is:", plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

