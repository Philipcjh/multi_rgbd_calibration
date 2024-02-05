import numpy as np
import yaml

def kabsch(P, Q):
    """
    Computes the optimal translation and rotation matrices that minimize the 
    RMS deviation between two sets of points P and Q using Kabsch's algorithm.
    More here: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Inspiration: https://github.com/charnley/rmsd
    
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q
            
    return: A 4 x 3 matrix where the first 3 rows are the rotation and the last is translation
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # 均值归一到0
    P_centered = P - centroid_P                       # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)                  # covariance matrix
    U, S, VT = np.linalg.svd(H)                        # SVD
    R = U.dot(VT).T                                    # calculate optimal rotation
    
    # 这里变换为右手系
    if np.linalg.det(R) < 0:                          # correct rotation matrix for             
        VT[2,:] *= -1                                  #  right-hand coordinate system
        R = U.dot(VT).T                          
    t = centroid_Q - R.dot(centroid_P)                # translation vector

    return R, t

def read_txt(file_path):
    with open(file_path, 'r') as file:
        # 逐行读取数据
        lines = file.readlines()
    
    # 解析数据并存储为 NumPy 数组
    data_array = np.array([list(map(float, line.strip().split())) for line in lines])
    return data_array


if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    camera_location1 = config["camera_location"]
    file1_path = config["data_path"] + camera_location1 + '/' + config["num_path"] + 'corners_points3d.txt'
    pts1 = read_txt(file1_path)
    
    camera_location2 = config["align_camera_location"]
    file2_path = config["data_path"] + camera_location2 + '/' + config["num_path"] + 'corners_points3d.txt'
    pts2 = read_txt(file2_path)

    # 从pts1变换到pts2
    R, t = kabsch(pts1, pts2)

    print("\nKabsch算法计算的旋转矩阵:")
    print(R)

    print("\nKabsch算法计算的平移向量:")
    print(t)
    
    print("\nKabsch算法计算的旋转矩阵")
    T = np.vstack([np.hstack((R, t.reshape(3, 1))), np.array([0, 0, 0, 1])])
    print(T)
    
    print("\nKabsch算法计算的旋转矩阵（直接复制到CloudCompare中）")
    formatted_output = '\n'.join(' '.join('{:0.12f}'.format(x) for x in row) for row in T)
    print(formatted_output)
    
    trans_txt = 'data/' + camera_location1 + '/' + camera_location1 + '2' + camera_location2 +'.txt'
    np.savetxt(trans_txt, T, fmt='%.12f', delimiter='\t')
    
