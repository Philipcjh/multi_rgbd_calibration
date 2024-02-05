import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

def depth_to_world(intr, depth):
    # 480 640 3
    height, width = depth.shape[:2]
    world = np.zeros((height, width, 3), dtype=np.float32)

    cx = intr[2]
    cy = intr[5]
    inv_fx = 1.0 / intr[0]
    inv_fy = 1.0 / intr[4]

    for r in range(height):
        for c in range(width):
            z = depth[r, c]
            if z == 0:
                world[r, c] = [np.nan, np.nan, np.nan]
            else:
                world[r, c] = [(c - cx) * z * inv_fx, (r - cy) * z * inv_fy, z]
    
    # # 创建一个掩码，表示非 NaN 值的位置
    # non_nan_mask = ~np.isnan(world)

    # # 通过掩码选择非 NaN 值
    # world_no_nan = np.where(non_nan_mask, world, 0)

    # print("Max Z:", np.nanmax(world_no_nan[:,:,2]))
    # print("Min Z:", np.nanmin(world_no_nan[:,:,2]))

    return world

def rearrange(corners, patterns):
    width, height = patterns[0], patterns[1]
    corners = corners.reshape(height, width, 2)
    # print(corners)
    
    # 行临点
    if corners[0, 0, 0] > corners[0, 1, 0]:
        left_to_right = False
    else:
        left_to_right = True
    
    # 列临点
    if corners[0, 0, 1] > corners[1, 0, 1]:
        top_to_bottom = False
    else:
        top_to_bottom = True
    
    if left_to_right and top_to_bottom:
        # left-top
        rearrange_corners = corners
    elif left_to_right and not top_to_bottom:
        # left-bottom
        rearrange_corners = np.flipud(corners)
    elif not left_to_right and top_to_bottom:
        # right-top
        rearrange_corners = np.fliplr(corners)
    else:
        # right-bottom
        rearrange_corners = np.flipud(np.fliplr(corners))
    rearrange_corners = rearrange_corners.reshape(-1, 1, 2)
    return rearrange_corners
    
def save_3d_points(poionts, filename):
    with open(filename, 'w') as file:
    # 遍历数据点并写入文件
        for point in poionts:
            # 使用字符串格式化将数据点按照 x y z 的格式写入文件
            file.write(f"{point[0]} {point[1]} {point[2]}\n")
    

if __name__ == "__main__":
    # 加载配置文件
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    camera_location = config["camera_location"]
    common_path = config["data_path"] + camera_location + '/' + config["num_path"]
    
    # 读取图像
    image_path = common_path + config["rgb_filename"]
    rgb_image = cv2.imread(image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 读取深度图
    depth_image_path = common_path + config["depth_filename"]
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    # 相机内参
    camera_resolution = config["camera_resolution"]
    intrinsics = config["camera_intrinsics"][camera_location][camera_resolution]
    
    # 还原到三维坐标
    world_coordinates = depth_to_world(intrinsics, depth_image)

    # 定义棋盘格的尺寸
    chessboard_size = (9, 6)  # 在这里，9 表示每行的角点数，6 表示每列的角点数

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(rgb_image, chessboard_size, None)
    

    if ret:
        # 重新排列，保证角点从左上角开始
        rearrange_corners = rearrange(corners, chessboard_size)
        # 四舍五入取整
        rearrange_corners = np.rint(rearrange_corners).astype(int)
        # print("Chessboard corners 2D coordinates:\n", rounded_corners)
        
        corners_points3d = []
        for index, corner in enumerate(rearrange_corners):
            # opencv坐标系：height-y, width-x
            x, y = corner.ravel()
            corners_point3d = [round(coord, 3) for coord in world_coordinates[y, x]]
            corners_points3d.append(corners_point3d)
            print(f'Point{index + 1}:', f'[u, v] = [{y}, {x}], ', " [x, y, z] =", corners_point3d)
            
        txt_path = common_path + config["corners_points_filename"]
        save_3d_points(corners_points3d, txt_path)
        print('3D points saved to', txt_path, 'successfully!!')

        # 在图像上标记角点
        cv2.drawChessboardCorners(rgb_image, chessboard_size, corners, ret)
        for index, corner in enumerate(rearrange_corners):
            # 标出起始角点
            if index == 0:
                x, y = corner.ravel()
                cv2.circle(rgb_image, (x, y), 3, (0, 255, 0), -1)  # 在每个角点处绘制绿色圆圈
                cv2.putText(rgb_image, f'({x},{y})', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 在每个角点处显示坐标
                break
        
        cv2.imwrite(common_path + config["corners_rgb_filename"], rgb_image)
        
        # 显示图像并自动调整窗口大小
        cv2.namedWindow('Chessboard Corners', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Chessboard Corners', rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Chessboard corners not found.")