import torchvision
from torch.utils.data import DataLoader


# Download training data from selected dataset
training_data = torchvision.datasets.STL10(
    root='datasets',
    split='train',
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Download test data from open datasets.
test_data = torchvision.datasets.STL10(
    root='datasets',
    split='test',
    download=True,
    transform=torchvision.transforms.ToTensor(),
) 

batch_size = 256

# Create training data loader
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# Create test data loader
test_dataloader = DataLoader(test_data, batch_size=batch_size)