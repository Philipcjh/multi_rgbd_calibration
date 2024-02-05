import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

if __name__ == "__main__":
    # 加载配置文件
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    common_path = config["data_path"] + config["camera_location"] + '/' + config["num_path"]
    
    # 读取RGB图像
    rgb_image = cv2.imread(common_path + config["rgb_filename"], cv2.IMREAD_UNCHANGED)

    # 读取深度图
    depth_image = cv2.imread(common_path + config["depth_align_filename"], cv2.IMREAD_UNCHANGED)

    # 如果深度图是单通道的，将其转换为三通道，以便与RGB图像重叠
    if len(depth_image.shape) == 2:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

    # 将深度图与RGB图像叠加
    overlay_image = cv2.addWeighted(rgb_image, 0.5, depth_image, 0.5, 0)

    # 使用Matplotlib可视化
    plt.figure(figsize=(10, 5))

    # 显示RGB图像
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('RGB Image')
    plt.axis('off')

    # 显示深度图像
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))
    plt.title('Depth Image')
    plt.axis('off')

    # 显示叠加图像
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay Image')
    plt.axis('off')
    
    # 保存图片
    plt.savefig(common_path + 'merge.png')
    
    plt.show()