import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def save_bgr_as_jpg(data, output_path, method='opencv'):
    bgr_data = data[:, :, :3] 
    if bgr_data.dtype != np.uint8:
        if bgr_data.max() > 1.0: 
            bgr_data = np.clip(bgr_data, 0, 255).astype(np.uint8)
        else:  
            bgr_data = (bgr_data * 255).clip(0, 255).astype(np.uint8)
    
    if method == 'opencv':
        cv2.imwrite(output_path, bgr_data)
        print(f"✅ OpenCV: {output_path}")
        
    elif method == 'pil':
        pil_image = Image.fromarray(bgr_data)
        pil_image.save(output_path, 'JPEG', quality=95)
        print(f"✅ PIL: {output_path}")
    
    return bgr_data

file_path = 'yieldpredicion/data/17007.npy'
data = np.load(file_path)
bgr_image = save_bgr_as_jpg(data[0,0,...], 'yieldpredicion/image_17007_0.jpg', method='opencv')
