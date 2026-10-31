# Dataset Information for FCE-Net

## Primary Dataset: LOLv1 (Low-Light paired dataset)

### Option A: Add from Kaggle Datasets (Recommended)
1. In your Kaggle notebook, click **"Add Data"** on the right panel
2. Search for: **"LOL dataset"** by Soumik Rakshit
3. Direct link: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
4. Click "Add" — it appears at `/kaggle/input/lol-dataset/`

### Option B: Upload from Google Drive
1. Download from the authors' link: https://drive.google.com/drive/folders/1tzf-wDJrhUvh415msACOPRzm518guz81
2. Upload as a new Kaggle Dataset
3. The notebook auto-detects the path

### Expected Structure
```
lol_dataset/
├── our485/          ← Training split (485 pairs)
│   ├── low/         ← Low-light input images
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ... (485 images)
│   └── high/        ← Normal-light ground truth images
│       ├── 1.png
│       ├── 2.png
│       └── ... (485 images)
└── eval15/          ← Evaluation split (15 pairs)
    ├── low/
    │   ├── 111.png
    │   └── ... (15 images)
    └── high/
        ├── 111.png
        └── ... (15 images)
```

### Image Properties
- Format: PNG
- Resolution: ~400×600 pixels
- Color: RGB
- Range: [0, 255] (loaded and normalized to [0, 1] by the dataloader)

### Pairing
- Low-light and normal-light images are paired by **sorted filename order**
- Each low-light image has exactly one corresponding normal-light ground truth

