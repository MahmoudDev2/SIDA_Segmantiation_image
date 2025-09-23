# Fake Image Detection Flask App

This project contains a Flask application to detect fake images using a pre-trained CNN model. It also includes a script for fine-tuning the model on a custom dataset.

## Quick Start

### 1. Download Pre-trained Weights (Required)

This application requires a pre-trained model file to run. Due to its size, you must download it manually.

1.  **Download the model:** [Click here to download from Dropbox](https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0)
2.  **Rename the file** to `blur_jpg_prob0.5.pth`.
3.  **Place the file** inside the `newflask/weights/` directory.

The application will not start correctly without this file.

### 2. Install Dependencies

```bash
# It is recommended to use a virtual environment
pip install -r requirements.txt
```
*(Note: A `requirements.txt` will be generated in a later step, for now you would need `torch`, `torchvision`, and `flask`)*

### 3. Run the Flask App

```bash
python newflask/app/app.py
```
Open your browser and go to `http://127.0.0.1:5000`. You can now upload an image to check if it's real or fake.

---

## Fine-Tuning the Model

You can fine-tune the detector on your own dataset.

### 1. Prepare Your Data

-   Place your **real** images in `newflask/training_data/full/real/`.
-   Place your **fake** images in `newflask/training_data/full/fake/`.

To test the script with a small number of images, you can use the `micro` directory:
-   Place a few **real** images in `newflask/training_data/micro/real/`.
-   Place a few **fake** images in `newflask/training_data/micro/fake/`.

### 2. Run the Fine-Tuning Script

To run a quick test on the "micro" dataset:
```bash
python newflask/finetune.py --data_dir newflask/training_data/micro --mode micro
```

To run a full training session:
```bash
python newflask/finetune.py --data_dir newflask/training_data/full --mode full --epochs 50 --lr 1e-5
```
The newly trained model will be saved in the `finetuned_models/` directory. You can then update the `MODEL_WEIGHT_PATH` in `app.py` to use your new model.

---

## Project Structure

-   **/app**: Contains the Flask web application (`app.py` and templates).
-   **/model_code**: Contains the Python source code for the image detection model and trainer.
-   **/weights**: Stores the pre-trained model weights.
-   **/training_data**: Contains directories for your custom datasets.
-   **finetune.py**: The script to fine-tune the detection model.
-   **localization_models_research.md**: A document with research on other models that support forgery localization.
