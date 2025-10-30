# BLIP Image Captioning Fine-Tuning on Custom Dataset
This repository demonstrates how to fine-tune the Salesforce BLIP (Bootstrapped Language Image Pretraining) model for image captioning on a custom dataset using PyTorch and Hugging Face Transformers in Google Colab.

It covers:
- Uploading custom image-caption datasets
- Building a PyTorch `Dataset` for image–text pairs
- Fine-tuning the BLIP model
- Generating captions for new images

---

## Overview
The BLIP model is a powerful image–text transformer capable of both caption generation and visual question answering.
In this project, we fine-tune it to generate more domain-specific captions using a small custom dataset.

## Requirements
Install all dependencies with:

```Python
!pip install -q torch torchvision transformers datasets pillow tqdm
```

This will install:\
PyTorch → for model training\
Transformers → for BLIP and text processing\
Datasets → optional Hugging Face utilities\
Pillow (PIL) → image loading\
tqdm → progress bars

---

## Dataset Format
You’ll need two inputs:\
1. A CSV file named captions.csv with at least two columns:

| image | caption |
| --- | ---: |
| image1.jpg | A cat sitting on a chair. |
| image2.jpg | A person riding a bicycle. |

A folder of images matching the filenames in the CSV.

2. A folder of images matching the filenames in the CSV.
```
project/
├── captions.csv
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

The notebook automatically uploads and stores the files under:

```Python
uploaded = files.upload()
df = pd.read_csv("captions.csv")
```

---

## Training Pipeline
1. Model and Processor Setup

```Python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
``` 

2. Dataset Class

A custom `CaptionDataset` pairs images with captions and prepares them for BLIP:

```Python
class CaptionDataset(Dataset):
    def __getitem__(self, idx):
        image = Image.open(f"{root_dir}/{row['image']}").convert("RGB")
        caption = row['caption']
        encoding = processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True)
```

3. Training Setup
- 80/20 train-validation split
- AdamW optimizer
- 2 epochs of training

```python
optimizer = AdamW(model.parameters(), lr=5e-5)
```

Each batch performs teacher-forced training:

```python
outputs = model(
    pixel_values=batch["pixel_values"],
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["input_ids"]
)
loss = outputs.loss
```
---

## Model Evaluation
After training, you can test the model on unseen images:

```python
test_image = Image.open("test_images/imageTest1.jpg").convert("RGB")
inputs = processor(images=test_image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(out[0], skip_special_tokens=True)
```

Example output:

```
Predicted caption: a small dog playing with a ball on the grass
```

---

## Usage Instructions
1. Open the notebook in Google Colab.
2. Upload your dataset files:
-  `captions.csv`
-  Your image set
3. Run the training cells.
4. Inspect the loss curve to confirm convergence.
5. Generate new captions using any image in `test_images/`.

## File Structure

```
.
├── captions.csv
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
├── test_images/
│   └── imageTest1.jpg
├── train_script.ipynb
└── README.md
```
