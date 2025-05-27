# 数据集准备
下载caltech-101数据集放在`./dataset`目录下，结构为：
```sh
./dataset/caltech-101/
├── 101_ObjectCategories/
└── Annotations/
```

# 模型准备
## 预训练模型权重
在根目录下运行：
```sh
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P weights/
```
下载在imagenet上预训练的resnet50的模型权重，用于微调训练。

## 最佳模型权重
从google drive https://drive.google.com/file/d/1oaoxORsqQ_wyWHX6VQv4cbLXRYzakU27/view?usp=sharing 下载最佳模型权重，放到weights目录下，命名为best_model.pth。这是已经微调过的模型权重，可直接用于caltech-101的测试。

# 训练
运行以下命令，使用默认超参数完成训练并测试：
```sh
python main.py --train
```

# 测试
运行以下命令，加载最佳模型权重，并单独进行测试：
```sh   
python main.py --test --ckp ./weights/best_model.pth
```