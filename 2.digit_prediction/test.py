import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

FORCE_RETRAIN = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="force retrain even if model.pth exists")
    args, _ = parser.parse_known_args()
    return args

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.fc4(x)
    
def get_data_loader(is_train = True):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    data_set = MNIST(root=data_root, train=is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=is_train)


def save_model(net, path: str):
    torch.save(net.state_dict(), path)


def load_model(net, path: str):
    state_dict = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict)

def evaluate(test_data, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net(x.view(-1, 28*28))
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

def main():

    args = parse_args()
    force_retrain = FORCE_RETRAIN or args.retrain

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    net = Net()

    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pth")
    has_model = os.path.exists(model_path)
    print(f"model_path: {model_path}")

    if has_model and not force_retrain:
        load_model(net, model_path)
        print("Model loaded successfully!")
    else:
        if force_retrain and has_model:
            print("Retraining and overwriting saved model (--retrain or FORCE_RETRAIN=True)...")
        else:
            print("No saved model found, training a new one!")
        optimize = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(10):
            net.train()
            for (x, y) in train_data:
                optimize.zero_grad()
                outputs = net(x.view(-1, 28*28))
                loss = torch.nn.functional.cross_entropy(outputs, y)
                loss.backward()
                optimize.step()
            print(f"Accuracy after epoch {epoch}:", evaluate(test_data, net))
        
        # 保存训练好的模型
        save_model(net, model_path)

    print("accuracy:", evaluate(test_data, net))

    for(n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        net.eval()
        with torch.no_grad():
            predict = net(x.view(-1, 28*28)).argmax(dim=1)[0].item()
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction:" + str(int(predict)))
    plt.show()

if __name__ == "__main__":
    main()
