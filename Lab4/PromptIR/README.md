## NYCU Computer Vision 2025 Spring Lab4

Student ID: 313551127

Name: 王翔 Hsiang, Wang

## Task Description

In Lab 4, we used **PromptIR** for an **all-in-one image restoration** task, which includes two degradation types: **derain** and **desnow**. The training dataset contains 1600 images for each degradation type. During testing, the specific degradation type of the input is **unknown**, and the model must rely on its prompt to infer the correct restoration approach. The evaluation metric is **PSNR**, where higher values indicate better performance. Additionally, the use of **external data** and **pretrained weights** is **not allowed**.

---

## Environment Setup

### 1. Clone the Repository

```sh
git clone https://github.com/Hsiungg/NYCU_CV.git
cd NYCU_CV/Lab4/PromptIR
```

---

### 2. Install Dependencies

#### Recommended Python Version

It is recommended to use **Python 3.10.17** for this lab.

Use the provided **env.yml** file from PrompIR to build up conda environment.

```sh
# Install required libraries and tools
conda env create -f env.yml
```


---

### 3. Download the Dataset

You can download the dataset from the following link:
[Download link](https://drive.google.com/drive/folders/1Q4qLPMCKdjn-iGgXV_8wujDmvDpSI1ul)

After downloading, place the **hw4-release_dataset.zip** file inside the cloned repository directory.

After placing the **hw4-release_dataset.zip**  file in the cloned directory, extract it using:

```sh
mkdir -p data
mv hw4-release_dataset.zip data/
cd data
unzip hw4-release_dataset.zip
cd ..
```

---

### 4. Set Up Training and Testing Environment

To start training the model, use the following command:

```sh
python train.py --training_root_dir "/path/to/training/img/directory" --ckpt_path "/path/to/output/ckpt/directory"
```

---

#### Training Options

- --training_root_dir: Path to the directory which put training images.
- --ckpt_path: Path to the directory for output model weight (.ckpt).

To start testing the model and output pred.npz file and all predicted clean images, use the following command:

```sh
python tester.py --output_path "/path/to/output/directory" --ckpt_path "/path/to/ckpt/directory" --ckpt_name "your_model_name.ckpt" --test_path "/path/to/test/images/directory"
```

#### Testing Options

- --output_path: Path of the directory for **pred.npz** file and all predicted clean images.
- --ckpt_path: Path of the directory which store .ckpt model weight.
- --model_name: The name of the model you want to test. ex: xxx.ckpt.
- --test_path: Path of the directory which store test images.
- --use_tta: If set, predict with TTA.

---

## Performance snapshots


![image](https://github.com/Hsiungg/NYCU_CV/blob/main/Lab3/final_result.png)

