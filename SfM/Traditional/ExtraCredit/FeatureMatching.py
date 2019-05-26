import cv2
import sys

sys.dont_write_bytecode = True


# img1 = cv2.imread('Images/1.jpeg')
# print(img1.shape)
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
# corners = np.int0(corners)

# for i in corners:
#     x, y = i.ravel()
#     cv2.circle(img1, (x, y), 7, 255, -1)

# plt.imshow(img1), plt.show()

f = open("matching1.txt", "w+")

img1 = cv2.imread("Images/1.jpeg")
img2 = cv2.imread("Images/2.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:50], None, flags=2)

for i in range(len(matches)):
    f.write(str(matches[i]))

# cv2.namedWindow('Img1', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Img1', 1000, 600)
# cv2.namedWindow('Img2', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Img2', 1000, 600)
cv2.namedWindow('Matching result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Matching result', 1000, 600)

# cv2.imshow("Img1", img1)
# cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
f.close()
