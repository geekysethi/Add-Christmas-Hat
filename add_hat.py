import numpy as np 
import cv2
from imutils import face_utils

import dlib

# image=cv2.imread('test2.jpg')
# image=cv2.resize(image,(400,400))



def add_hat(image):


	hat_img = cv2.imread ("hat2.png", - 1)
	r, g, b, a = cv2.split (hat_img)
	rgb_hat = cv2.merge ((r, g, b))

	detector=dlib.get_frontal_face_detector()
	predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	rects=detector(image,1)

	for (i,rect) in enumerate(rects):

		shape=predictor(image,rect)
		shape=face_utils.shape_to_np(shape)
		
		print(len(shape))

		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	 
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


		# for (x, y) in shape:
		# 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		# print(shape[0])
		point1=shape[19]
		point2=shape[24]
		eyes_center = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

		cv2.circle (image, eyes_center, 3, color = (0,255,0))
		# cv2.imshow ("image", image)
		# cv2.waitKey ()

		factor = 1.5
		print(h,w)
		print(rgb_hat.shape)
		# resized_hat_h = int (round (h * factor))
		# resized_hat_w = int (round (w * factor))
		resized_hat_h = int (round (rgb_hat.shape [0] * w / rgb_hat.shape [1] * factor))
		resized_hat_w = int (round (rgb_hat.shape [1] * w / rgb_hat.shape [1] * factor))

		resized_hat = cv2.resize (rgb_hat, (resized_hat_w, resized_hat_h))
		
		# cv2.imshow ("resized_hat", resized_hat)
		# cv2.waitKey ()

		if resized_hat_h> y:
			resized_hat_h = y-1

		print(resized_hat_h,resized_hat_w)

		mask = cv2.resize (a, (resized_hat_w, resized_hat_h))
		mask_inv = cv2.bitwise_not (mask)
		# cv2.imshow ("mask", mask)
		# cv2.waitKey ()

		# print(x,y)
		dh = 0
		dw = 0
		# Original ROI
		bg_roi = image [y + dh-resized_hat_h: y + dh, (eyes_center [0] -resized_hat_w // 3) :( eyes_center [0] + resized_hat_w // 3 * 2)]

		# print(bg_roi.shape)
		# cv2.imshow ("bg_roi", bg_roi)
		# cv2.waitKey ()


		bg_roi = bg_roi.astype (float)
		mask_inv = cv2.merge ((mask_inv, mask_inv, mask_inv))
		alpha = mask_inv.astype (float) / 255

		# Before multiplying to ensure that both the same size (may not be consistent due to rounding)
		alpha = cv2.resize (alpha, (bg_roi.shape [1], bg_roi.shape [0]))
		# print ("alpha size:", alpha.shape)
		# print ("bg_roi size:", bg_roi.shape)
		bg = cv2.multiply (alpha, bg_roi)
		bg = bg.astype ('uint8')

		# cv2.imwrite ("bg.jpg", bg)
		# cv2.imshow ("bg", bg)
		# cv2.waitKey ()

		# Extract hat area
		# resized_hat=resized_hat.astype('uint8')
		# print(resized_hat.shape)
		# print(mask.shape)
		mask=cv2.resize(mask,(resized_hat.shape[1],resized_hat.shape[0]))
		hat = cv2.bitwise_and (resized_hat, resized_hat, mask = mask)
		# cv2.imwrite ("hat.jpg", hat)
				
		# cv2.imshow ("hat", hat)
		# cv2.imshow ("bg", bg)

		# print ("bg size:", bg.shape)
		# print ("hat size:", hat.shape)

		# Before adding to ensure that both the same size (may not be consistent due to rounding)
		hat = cv2.resize (hat, (bg_roi.shape [1], bg_roi.shape [0]))

		# Add two ROI regions
		add_hat = cv2.add (bg, hat)
		# cv2.imshow ("add_hat", hat)

		# Put the hat added area back to the original picture
		image [y + dh-resized_hat_h: y + dh, (eyes_center [0] -resized_hat_w // 3) :( eyes_center [0] + resized_hat_w // 3 * 2)] = add_hat

	return image
 	# Display of results
	# cv2.imshow ("img", img)
	# cv2.waitKey (0)

		
	




# image=add_hat(image)

# cv2.imshow("Output", image)
# cv2.waitKey(0)
