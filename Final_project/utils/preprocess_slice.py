import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import product


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

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def produce_seg_npy(df, output_dir, image_dir="sartorius-cell-instance-segmentation/train", save_image=True):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby('id')
    for image_id, group in tqdm(grouped, desc="Generating masks"):
        height = group.iloc[0]['height']
        width = group.iloc[0]['width']
        rle_list = group['annotation'].tolist()  # 所有該 image_id 的 RLE
        img_path = os.path.join(image_dir, f"{image_id}.png")
        img = cv2.imread(img_path)

        H, W = img.shape[:2]

        # ms = []
        # for mask in rle_list:
        #     mask = np.asfortranarray(enc2mask([mask], (height, width)))
        #     ms.append(mask)
        ms = enc2mask(rle_list, (height, width))
        # ms_sum = ms.sum((0,1))
        # print(f"ms_sum shape: {ms_sum.shape}")
        cuts = 4
        wstarts = W * np.arange(cuts).astype(int) // (cuts + 1)
        wends = W * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
        hstarts = H * np.arange(cuts).astype(int) // (cuts + 1)
        hends = H * np.arange(2, cuts + 2).astype(int) // (cuts + 1)
        for i, j in product(range(cuts), range(cuts)):
            
            img_cut = img[hstarts[i]:hends[i],wstarts[j]:wends[j]]
            mask_cut = ms[hstarts[i]:hends[i],wstarts[j]:wends[j]]
            # mask_cut = mask_cut[...,mask_cut.sum((0,1)) > 0.25 * ms_sum]
            cv2.imwrite(os.path.join(output_dir, f"{image_id}_{i}_{j}.jpg"), img_cut)
            
            # 儲存為 .npy
            out_path = os.path.join(output_dir, f"{image_id}_{i}_{j}_seg.npy")
            # 將 mask 包成 dict
            mask_dict = {'masks': mask_cut}
            with open(out_path, 'wb') as f:
                np.save(f, mask_dict, allow_pickle=True)


if __name__ == "__main__":
    # 設定來源資料夾
    file_names = glob.glob("sartorius-cell-instance-segmentation/train/*.png")
    labels = pd.read_csv("sartorius-cell-instance-segmentation/train.csv")
    image_dir = "sartorius-cell-instance-segmentation/train"

    # 儲存到哪個資料夾
    train_dir = "sartorius-cell-instance-segmentation/train_tiny"
    val_dir = "sartorius-cell-instance-segmentation/val_tiny"

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