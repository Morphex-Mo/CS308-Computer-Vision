import pickle
import numpy as np

def save_matches_as_pkl(x1, y1, x2, y2, matches, pkl_path):
    """
    将自动匹配的结果保存为与 CorrespondenceAnnotator 生成的 pkl 相同的格式。
    
    参数:
        x1, y1, x2, y2: 兴趣点坐标数组 (来自 get_interest_points)
        matches: (k,2) 匹配索引，第一列对应 img1 的索引，第二列对应 img2 的索引
        pkl_path: 输出 pkl 文件路径（例如 'mount_rushmore_correspondences.pkl'）
    """
    # 提取匹配点的实际坐标
    matched_x1 = x1[matches[:, 0]].tolist()
    matched_y1 = y1[matches[:, 0]].tolist()
    matched_x2 = x2[matches[:, 1]].tolist()
    matched_y2 = y2[matches[:, 1]].tolist()
    
    data_dict = {
        'x1': matched_x1,
        'y1': matched_y1,
        'x2': matched_x2,
        'y2': matched_y2
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved {len(matched_x1)} matches to {pkl_path}")