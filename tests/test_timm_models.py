# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from tqdm import tqdm
# import timm
# import torchvision.transforms as T
# from torch.cuda import amp
# from vision.datasets import create_cifar10_dataset, create_loaders
# from vision.models.classification import model_factory
# from vision.models.classification import engine
# # from torch.optim.swa_utils import SWALR

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def test_models():

#     MODEL_NAME = "efficientnet_b0"  # For now a very small model

#     print("Creating Train and Validation Dataset")
#     train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
#     valid_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

#     # Test defaults
#     train_set, valid_set = create_cifar10_dataset(train_transforms, valid_transforms)
#     print("Train and Validation Datasets Created")

#     print("Creating DataLoaders")
#     train_loader, valid_loader = create_loaders(train_set, train_set)

#     print("Train and Validation Dataloaders Created")
#     print("Creating Model")

#     # Right method and complete check would be
#     # for model in timm.list_models() and then do this. It willl go out of github actions limits.
#     model = model_factory.create_timm_cnn(MODEL_NAME, num_classes=10,
#                                           in_channels=3, pretrained=False,)

#     if torch.cuda.is_available():
#         print("Model Created. Moving it to CUDA")
#     else:
#         print("Model Created. Training on CPU only")
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     criterion = nn.CrossEntropyLoss()  # All classification problems we need Cross entropy loss

#     # early_stopper = utils.EarlyStopping(
#     #     patience=7, verbose=True, path=SAVE_PATH
#     # )
#     # We do not need early stopping too

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

#     # swa_scheduler = SWALR(optimizer, anneal_strategy="linear",
#     #                       anneal_epochs=20, swa_lr=0.05)
#     swa_start = 2

#     if torch.cuda.is_available():
#         scaler = amp.GradScaler()

#         train_metrics = engine.train_step(model, train_loader, criterion,
#                                           device, optimizer,
#                                           num_batches=10,
#                                           fp16_scaler=scaler,)

#         history2 = engine.fit(1, model, train_loader, valid_loader,
#                               criterion, device, optimizer, num_batches=10,
#                               grad_penalty=True, use_fp16=True,)

#     train_metrics = engine.train_step(model, train_loader,
#                                       criterion, device, optimizer,
#                                       num_batches=10,)

#     history = engine.sanity_fit(model, train_loader, valid_loader,
#                                 criterion, device, num_batches=10,
#                                 grad_penalty=True,)

#     history2 = engine.fit(1, model, train_loader,
#                           valid_loader, criterion,
#                           device, optimizer,
#                           num_batches=10, grad_penalty=True,)

#     # history3 = engine.fit(3, model, train_loader, valid_loader, criterion,
#     #                       device, optimizer, scheduler=scheduler,
#     #                       num_batches=10, grad_penalty=True,
#     #                       swa_start=swa_start, swa_scheduler=swa_scheduler,)

#     print("Done !!")
#     return 1
