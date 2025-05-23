# NYCU Computer Vision 2025 Spring Lab2

Student ID: 313551127

Name: 王翔 Hsiang, Wang

## Task Description

Lab 2 uses **Fast R-CNN** to perform **object detection** on images containing digits. It is divided into two tasks: identifying the bounding box of each digit, and recognizing which digits are present in the entire image. The performance is evaluated based on the **mAP** and **Accuracy**.

---

## Environment Setup

### 1. Clone the Repository

```sh
git clone https://github.com/Hsiungg/NYCU_CV.git
cd NYCU_CV/Lab2
```

---

### 2. Install Dependencies

#### Recommended Python Version

It is recommended to use **Python 3.12.9** for this lab.

Use the provided **requirements.txt** to handle the environment package dependencies.

```sh
# Install required libraries and tools
pip install -r requirements.txt
```

---

### 3. Download the Dataset

You can download the dataset from the following link:
[Download link](<https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view>)

After downloading, place the **nycu-hw2-data.tar.gz** file inside the cloned repository directory.

```sh
tar -xzvf nycu-hw2-data.tar.gz
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

To start testing the model and output **pred.csv** and **pred.json**file, use the following command:

```sh
python tester.py --save_dir /path/to/save/directory
```

#### Testing Options

- --save_dir: Path of the save directory for **output_config.yaml** file and model.pth file.
- --model_name: The name of the model you want to test. ex: xxx.pth

---

## Performance snapshots

![image](https://github.com/Hsiungg/NYCU_CV/blob/main/Lab2/final_score.png)
