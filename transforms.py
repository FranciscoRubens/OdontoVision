import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transform():
    return A.Compose([
        A.Resize(256, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])