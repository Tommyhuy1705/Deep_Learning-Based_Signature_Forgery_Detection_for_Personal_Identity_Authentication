│
├── models/
│   ├── feature_extractor.py        # Chứa CNN ResNet34 
│   ├── triplet_network.py           # TripletNetwork class
│   ├── siamese_network.py           # SiameseNetwork class
│
├── losses/
│   ├── triplet_loss.py              # Hàm loss cho Triplet Network
│   ├── contrastive_loss.py          # Hàm loss cho Siamese Network
│
├── datasets/
│   ├── triplet_dataset_loader.py           # Dataset class cho Triplet loader, load data cho train
│   ├── siamese_dataset_loader.py           # Dataset class cho Siamese loader, load data cho train
│
├── utils/
│   ├── metrics.py                   # accuracy, distance, F1, ....
│   ├── helpers.py                   # Hàm hỗ trợ cho việc train dễ hơn (ví dụ: load checkpoint, vísualize)
│
├── configs/
│   ├── config_triplet.yaml          # File cấu hình riêng cho Triplet
│   ├── config_siamese.yaml          # File cấu hình riêng cho Siamese
│
├── notebooks/
│   ├── train_triplet.ipynb          # Train Triplet
│   ├── train_siamese.ipynb          # Train Siamese
│
├── README.md                        
└── requirements.txt                 
