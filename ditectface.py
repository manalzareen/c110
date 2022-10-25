# import the opencv library
import cv2
import tensorflow as tf 
import numpy as np

model = tf.keras.models.load_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    
      
     # 1. Resizing the image
    img = cv2.resize(frame,(224,224))

     # 2. Converting the image into Numpy array and increase dimension
    testimage = np.array(img,dtype=np.float32)
    testimage = np.expand_dims(testimage , axis = 0)

    # 3. Normalizing the image
    n_image = testimage/255.0
    p = model.predict(n_image)
    print("prediction : ", p )


    cv2.imshow('frame', frame)

    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()

#prediction : [[0.745333 0.22455]]