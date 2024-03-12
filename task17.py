import cv2
from vidgear.gears import CamGear
import time

class Camera1():
   def __init__(self):
       super(Camera1, self).__init__()
       self.previousFrame = None

   def run(self):
       stream = CamGear(source=0).start()

       while(True):
           start_time = time.time()
           frame = stream.read()
           if self.previousFrame is not None:
               composicion = cv2.addWeighted(frame, 0.8, self.previousFrame, 0.2, 0.0)
           else:
               self.previousFrame = frame
               composicion = cv2.addWeighted(frame, 0.8, self.previousFrame, 0.2, 0.0)
           cv2.imshow('Video', composicion)
           self.previousFrame = composicion
           end_time = time.time()
           print("Время выполнения: ", end_time - start_time)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

       cv2.destroyAllWindows()

if __name__ == '__main__':
   c = Camera1()
   c.run() 