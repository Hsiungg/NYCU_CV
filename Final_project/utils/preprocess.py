import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ref : https://github.com/CarnoZhao/mmdetection/blob/sartorius_solution/sartorius/data.ipynb
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype = np.uint16)
    for m, enc in enumerate(encs):
        if isinstance(enc, float) and np.isnan(enc):
            continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m # 1-based indexing
    return img.reshape(shape)

def produce_seg_npy(df, output_dir, image_dir="sartorius-cell-instance-segmentation/train", save_image=True):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby('id')
    for image_id, group in tqdm(grouped, desc="Generating masks"):
        height = group.iloc[0]['height']
        width = group.iloc[0]['width']
        rle_list = group['annotation'].tolist()  # 所有該 image_id 的 RLE

        full_mask = enc2mask(rle_list, (height, width))
        print(f"full_mask shape: {full_mask.shape}")
        # 將 mask 包成 dict
        mask_dict = {'masks': full_mask}

        # 儲存為 .npy
        out_path = os.path.join(output_dir, f"{image_id}_seg.npy")
        with open(out_path, 'wb') as f:
            np.save(f, mask_dict, allow_pickle=True)

        if save_image:
            img_path = os.path.join(image_dir, f"{image_id}.png")
            img = cv2.imread(img_path)
            if img is not None:
                out_img_path = os.path.join(output_dir, f"{image_id}.png")
                cv2.imwrite(out_img_path, img)
            else:
                print(f"[警告] 無法讀取圖片 {img_path}")


if __name__ == "__main__":
    # 設定來源資料夾
    file_names = glob.glob("sartorius-cell-instance-segmentation/train/*.png")
    labels = pd.read_csv("sartorius-cell-instance-segmentation/train.csv")
    image_dir = "sartorius-cell-instance-segmentation/train"

    # 儲存到哪個資料夾
    train_dir = "sartorius-cell-instance-segmentation/mytrain"
    val_dir = "sartorius-cell-instance-segmentation/myval"

    # validation set 的比例
    val_size = 0.2


    # 為每張圖片選定一個 cell_type（多數類別代表該圖片的 cell_type）
    image_info = labels.groupby("id").agg({
        "cell_type": lambda x: x.mode()[0],  # 取出最多的 cell_type
        "width": "first",
        "height": "first"
    }).reset_index()

    # 分層切割器
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

    # 執行分割
    for train_idx, val_idx in splitter.split(image_info["id"], image_info["cell_type"]):
        train_ids = image_info.loc[train_idx, "id"].values
        val_ids = image_info.loc[val_idx, "id"].values

    # 建立訓練與驗證集的 DataFrame
    train_df = labels[labels["id"].isin(train_ids)].reset_index(drop=True)
    val_df = labels[labels["id"].isin(val_ids)].reset_index(drop=True)

    # 印出統計資訊
    print("Train images:", train_df["id"].nunique())
    print("Val images:", val_df["id"].nunique())
    print("Cell types in train:")
    print(train_df["cell_type"].value_counts())
    print("Cell types in val:")
    print(val_df["cell_type"].value_counts())


    produce_seg_npy(train_df, train_dir, image_dir)
    produce_seg_npy(val_df, val_dir, image_dir)