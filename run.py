import test,cv2,numpy as np, matplotlib.pyplot as plt
c,d,i = test.load('withdog')
cc,dd,ii = test.load('background')
du = test.to_uint8(d)
ddu = test.to_uint8(dd)

subtracted = cv2.subtract(du,ddu)

subtracted = cv2.medianBlur(subtracted,7)
kernel = np.ones((3,3),np.uint8)
subtracted = cv2.morphologyEx(subtracted, cv2.MORPH_OPEN, kernel)

ret,subtracted = cv2.threshold(subtracted, 1, 255, cv2.THRESH_BINARY)

image, cnts, hierarchy = cv2.findContours(subtracted.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in cnts if cv2.contourArea(c) >= 1000]
(x, y, width, height) = cv2.boundingRect(contours[0])
dogdepth = d[y:y+height, x:x+width]
dogred = i[y:y+height, x:x+width]

plt.subplot(121).imshow(dogdepth, 'gray')
plt.subplot(122).imshow(dogred, 'Reds')
plt.show()

dogred_thresh = test.to_uint8(dogred)
thresholdVal = int(np.mean([dogred_thresh[0],dogred_thresh[-1]]))+1
ret,dogred_thresh = cv2.threshold(dogred_thresh, thresholdVal, 255, cv2.THRESH_BINARY)
test.show(dogred_thresh,'Reds')

mask = np.uint16(dogred_thresh) * 257
depth = np.bitwise_and(65535-d[y:y+height, x:x+width],mask)

### Don't forget that at this point, depth is now back in MM.

depth_trans = np.transpose(depth)
stride = int(depth_trans.shape[0]/10)
for r in range(0,depth_trans.shape[0]-stride,stride):
    total = np.sum(depth_trans[r:r+stride])
    count = np.count_nonzero(depth_trans[r:r+stride])
    print(total/count)