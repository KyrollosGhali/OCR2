# import csv
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt


class BubbleSheetScanner:
    def __init__(self, bubble_count=5):
        self.bubbleCount = bubble_count
        self.bubbleWidthAvr = 0
        self.bubbleHeightAvr = 0

    def getCannyFrame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 127, 255)

    def getAdaptiveThresh(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 7
        )

    def getFourPoints(self, canny):
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []

        for cnt in contours:
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if len(approx) == 4 and 0.9 <= aspect_ratio <= 1.1:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))

        return points

    def getWarpedFrame(self, cannyFrame, frame):
        points = self.getFourPoints(cannyFrame)
        if len(points) < 4:
            return None

        points = np.array(points, dtype="float32")
        selected = [points[0], points[1], points[-2], points[-1]]
        return four_point_transform(frame, np.array(selected, dtype="float32"))

    def getOvalContours(self, adaptiveFrame):
        contours, _ = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovalContours = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue

            aspect_ratio = w / float(h)
            approx = cv2.approxPolyDP(c, 0, True)

            if len(approx) > 15 and 0.8 <= aspect_ratio <= 1.2:
                ovalContours.append(c)
                self.bubbleWidthAvr += w
                self.bubbleHeightAvr += h

        if ovalContours:
            self.bubbleWidthAvr /= len(ovalContours)
            self.bubbleHeightAvr /= len(ovalContours)

        return ovalContours

    def x_cord(self, c):
        x, y, w, h = cv2.boundingRect(c)
        return y + x * self.bubbleHeightAvr

    def y_cord(self, c):
        x, y, w, h = cv2.boundingRect(c)
        return x + y * self.bubbleWidthAvr


# =======================
# MAIN
# =======================

# scanner = BubbleSheetScanner(bubble_count=5)

# image = cv2.imread("optik11.jpg")
# h = int(600 * image.shape[0] / image.shape[1])
# frame = cv2.resize(image, (600, h))

# canny = scanner.getCannyFrame(frame)
# warped = scanner.getWarpedFrame(canny, frame)

# if warped is None:
#     print("Sheet not detected")
#     exit()

# adaptive = scanner.getAdaptiveThresh(warped)
# ovals = scanner.getOvalContours(adaptive)

# total_bubbles = len(ovals)
# # if total_bubbles % scanner.bubbleCount != 0:
# #     print("Invalid bubble layout")
# #     exit()

# questionCount = total_bubbles // scanner.bubbleCount
# print(f"Detected questions: {questionCount}")

# ovals = sorted(ovals, key=scanner.y_cord)

# marked_answers = []
# fill_threshold = 1.0

# for q in range(questionCount):
#     start = q * scanner.bubbleCount
#     bubbles = sorted(
#         ovals[start:start + scanner.bubbleCount],
#         key=scanner.x_cord
#     )

#     best_index = None
#     best_ratio = 0.0

#     for j, c in enumerate(bubbles):
#         area = cv2.contourArea(c)
#         mask = np.zeros(adaptive.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)
#         masked = cv2.bitwise_and(adaptive, adaptive, mask=mask)
#         filled = cv2.countNonZero(masked)

#         ratio = filled / area if area > 0 else 0
#         if ratio > best_ratio:
#             best_ratio = ratio
#             best_index = j

#     if best_ratio > fill_threshold:
#         marked_answers.append(best_index)
#         cv2.drawContours(warped, bubbles, best_index, (0, 255, 0), 2)
#     else:
#         marked_answers.append(None)

# plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

# # CSV OUTPUT
# with open("marked_answers.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Question", "Marked Choice"])
#     for i, ans in enumerate(marked_answers, 1):
#         writer.writerow([i, "" if ans is None else ans])

# print("Saved to marked_answers.csv")
