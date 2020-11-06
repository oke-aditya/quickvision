# import pytorch_lightning as pl
# from vision.datasets import create_cifar10_dataset, create_loaders
# import torchvision.transforms as T
# from vision.models.classification.lightning_trainer import lit_cnn


# def test_lit_trainer():
#     pl.seed_everything(seed=42)

#     print("Creating Train and Validation Dataset")
#     train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
#     valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

#     train_set, valid_set = create_cifar10_dataset(train_transforms, valid_transforms)
#     print("Train and Validation Datasets Created")

#     print("Creating DataLoaders")
#     train_loader, valid_loader = create_loaders(train_set, train_set)

#     print("Train and Validation Dataloaders Created")
#     print("Creating Model")

#     print("Train and Validation Dataloaders Created")

#     print("Creating Model")
#     all_supported_models = [
#         "resnet18",  # "resnet34", # "resnet50", # "resnet101", # "resnet152",
#         # "resnext50_32x4d", # "resnext101_32x8d", # "vgg11",
#         # "vgg13", # "vgg16", # "vgg19", # "mobilenet", # "mnasnet0_5",
#         # "mnasnet1_0",
#     ]

#     for model_name in all_supported_models:
#         model = lit_cnn(model_name, num_classes=10, pretrained=True)
#         # print(model)
#         trainer = pl.Trainer(fast_dev_run=True)
#         trainer.fit(model, train_loader, valid_loader)

#     return True
