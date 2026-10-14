import cv2

print("Đang dò tìm camera...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"-> Camera cổng số {i} đang hoạt động!")
        cap.release()