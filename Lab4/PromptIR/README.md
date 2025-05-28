## NYCU Computer Vision 2025 Spring Lab3

Student ID: 313551127

Name: 王翔 Hsiang, Wang

## Task Description

In Lab 3, **Mask R-CNN** was used to perform instance segmentation on a dataset containing four types of cells. The model size (trainable parameters) was constrained to a maximum of 200 M, and only the **Mask R-CNN** architecture was allowed. The final evaluation metric was the segmentation **mAP@50**.

---

## Environment Setup

### 1. Clone the Repository

```sh
git clone https://github.com/Hsiungg/NYCU_CV.git
cd NYCU_CV/Lab3
```

---

### 2. Install Dependencies

#### Recommended Python Version

It is recommended to use **Python 3.10.17** for this lab.

Use the provided **requirements.txt** to handle the environment package dependencies.

```sh
# Install required libraries and tools
pip install -r requirements.txt
```

If this failed (third party package from github sometimes need to be installed manually), try to install **Detectron2** from [here](<https://github.com/facebookresearch/detectron2?tab=readme-ov-file#installation>) by following the **Installation** part.

---

### 3. Download the Dataset

You can download the dataset from the following link:
[Download link](<https://drive.google.com/file/d/1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI/view>)

After downloading, place the **hw3-data-release.tar.gz** file inside the cloned repository directory.

After placing the dataset.tar.gz file in the cloned directory, extract it using:

```sh
mkdir -p data
mv hw3-data-release.tar.gz data/
cd data
tar -xzvf hw3-data-release.tar.gz
cd ..
```

---

### 4. Set Up Training and Testing Environment

To start training the model, use the following command:

```sh
python trainer.py --output_dir /path/to/output/directory
```

---

#### Training Options

- --output_dir: Path to the output directory for model weights and model yaml.

To start testing the model and output **test-results.json** file, use the following command:

```sh
python tester.py --save_dir /path/to/save/directory --model_name "your_model_name.pth"
```

#### Testing Options

- --save_dir: Path of the save directory for **output_config.yaml** file and model.pth file.
- --model_name: The name of the model you want to test. ex: xxx.pth

---

## Performance snapshots


![image](https://github.com/Hsiungg/NYCU_CV/blob/main/Lab3/final_result.png)

