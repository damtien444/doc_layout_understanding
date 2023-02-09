import numpy as np
from PIL import Image

image_path = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_selected_sample/Loại 1_GDCD_Thi Thử THPTQG 2021__1._Đề_thi_thử_THPTQG_2021_-_GDCD_-_THPT_Lý_Thái_Tổ_-_Bắc_Ninh_-_Lần_1-_File_word_có_đáp_án__1.png"
img = Image.open(image_path)
width = img.width
height = img.height
img = np.array(img)
print(img)
print(width, height)