#%%
import cv2
import numpy as np

def mask_sebum(img_path:str, 
               porphyrin_r_tres:int=200, 
               porphyrin_g_tres:int=200,
               sebum_r_tres:int=160,  
               sebum_g_tres:int=190,  
               sebum_b_tres:int=210):
    """
    UV LED로 촬영한 사진에서 피지가 있는 부분을\n
    마스킹하여 강조한 이미지의 배열과 마스킹한 부위가 차지하는 비율을 반환합니다.\n
    ---
    ## Parameters:  
        img_path (str): 이미지의 경로  
        porphyrin_r_tres (int): 포피린 마스크 R채널에 대한 임계값\n
        porphyrin_g_tres (int): 포피린 마스크 G채널에 대한 임계값\n
        sebum_r_tres (int): 일반 피지 마스크 R채널에 대한 임계값\n
        sebum_g_tres (int): 일반 피지 마스크 G채널에 대한 임계값\n
        sebum_b_tres (int): 일반 피지 마스크 B채널에 대한 임계값\n
      
    ## Returns:  
        output_array (ndarray): 원본 이미지의 배열과 마스킹의 배열을 합성한 배열\n 
        ratio (float): 전체 면적 (890*890)중 마스킹된 부분의 비율  
    """
    img_array = cv2.imread(img_path)
    lab_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)

    red_array = img_array[..., 2]
    green_array = img_array[..., 1]
    blue_array = img_array[..., 0]
    
    l_array = lab_array[..., 0]
    a_array = lab_array[..., 1]
    b_array = lab_array[..., 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_array = clahe.apply(l_array)
    lab_array = cv2.merge((l_array, a_array, b_array))
    modified_rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)

    modified_red_array = modified_rgb_array[..., 0]
    modified_green_array = modified_rgb_array[..., 1]
    modified_blue_array = modified_rgb_array[..., 2]

    # 포피린 마스크
    porphyrin_mask_array = np.zeros_like(red_array, dtype=bool)
    porphyrin_mask_array[modified_red_array > porphyrin_r_tres] = True
    porphyrin_mask_array[modified_red_array <= porphyrin_r_tres] = False
    porphyrin_mask_array[modified_green_array > porphyrin_g_tres] = False
    
    # 일반 피지 마스크
    sebum_mask_array = np.zeros_like(porphyrin_mask_array, dtype=bool)
    sebum_mask_array[(modified_red_array > sebum_r_tres) &
                     (modified_green_array > sebum_g_tres) &
                     (modified_blue_array > sebum_b_tres)] = True

    # 포피린 마스킹 색상
    red_array[porphyrin_mask_array] = 255
    green_array[porphyrin_mask_array] = 60
    blue_array[porphyrin_mask_array] = 60

    # 일반 피지 마스킹 색상
    red_array[sebum_mask_array] = 50
    green_array[sebum_mask_array] = 255
    blue_array[sebum_mask_array] = 130

    # 마스킹된 부분의 면적 계산
    height, width = porphyrin_mask_array.shape
    total_area = height * width
    porphyrin_area = porphyrin_mask_array.sum()
    porphyrin_ratio = porphyrin_area / total_area
    sebum_area = sebum_mask_array.sum()
    sebum_ratio = sebum_area / total_area
    
    output_array = cv2.merge((red_array, green_array, blue_array))
    return output_array, porphyrin_ratio, sebum_ratio

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    ROOT_PATH = "../dataset/skin/uv"
    imgs_list = os.listdir(ROOT_PATH)
    for i in range(30,100):
        if str.endswith(imgs_list[i], '.png'):
            img_path = os.path.join(ROOT_PATH, imgs_list[i])
            print(imgs_list[i])
            masked_array, *ratio = mask_sebum(img_path)
            print(ratio)
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.title('original')
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.subplot(122)
            plt.title('masked')
            plt.imshow(masked_array)
            plt.show()
        else:
            continue



# %%
