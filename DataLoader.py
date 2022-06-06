
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
      
    
def loadData(TrainImagesPath, TestImagesPath):
    transformation = transforms.Compose( [transforms.Resize((32,32)), 
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5])])
    # dataset = torchvision.datasets.ImageFolder(root=TrainImagesPath, transform=transformation)
    
    testSet = torchvision.datasets.ImageFolder(root=TestImagesPath, transform=transformation)
    trainSet = torchvision.datasets.ImageFolder(root=TrainImagesPath, transform=transformation)
    
    # print(testSet.classes)
    # print(trainSet.classes)
    
    # train_loader = torch.utils.data.random_split(len(trainSet))
    # test_loader = torch.utils.data.random_split(len(testSet))
    
    train_loader = torch.utils.data.DataLoader(trainSet,batch_size=5,num_workers=2,shuffle=False)
    test_loader = torch.utils.data.DataLoader(testSet,batch_size=5,num_workers=2,shuffle=False)
    
    return train_loader, test_loader


# Resize the images so that they are the same sizes (manually, should not be needed)
def normalizeResolution(imagesPath, resolution):
    dirs = os.listdir(imagesPath )
    for item in dirs:
        if os.path.isfile(imagesPath+item):
            im = Image.open(imagesPath+item)
            f, e = os.path.splitext(imagesPath+item)
            imResize = im.resize(resolution, Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=100)
    return 0


#print(loadData("./TrainSet","./TestSet"))




















