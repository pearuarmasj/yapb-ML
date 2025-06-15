import mss
import numpy as np
import cv2

def grab_screen_and_save(filename='screenshot.png'):
    """Grab the screen and save it as an image file."""
    with mss.mss() as sct:
        # Grab the entire screen
        screenshot = sct.grab(sct.monitors[1])
        
        # Convert to numpy array
        img = np.array(screenshot)

        img = img[..., :3]  # Remove alpha channel if present
        
        # Convert BGRA to BGR (OpenCV uses BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Save the image
        cv2.imwrite(filename, img)
        print(f"Screenshot saved as {filename}")

grab_screen_and_save('screenshot.png')

cv2.imshow('Screenshot', cv2.imread('screenshot.png'))
cv2.waitKey(0)
cv2.destroyAllWindows()