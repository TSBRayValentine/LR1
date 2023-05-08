
from CheckpointSaver import CheckpointSaver
from Tools import Precision, Recall
from NeuralNetwork import NeuralNetwork

from FashionMNISTDataset import FashionMNISTDataset
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

# Конфигурация
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig):

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    def plot_samples(data, labels_map, cols=3, rows=3):
        figure = plt.figure(figsize=(8, 8))
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(data), size=(1,)).item()
            img, label = data[sample_idx]
            figure.add_subplot(rows, cols, i)
            if type(label) == torch.Tensor:
                plt.title(labels_map[label.item()])
            else:
                plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
        plt.show()

    plot_samples(train_data, labels_map, cols=3, rows=3)

    train_data = FashionMNISTDataset(
        "data/FashionMNIST/raw", train=True)  # ваш код
    test_data = FashionMNISTDataset(
        "data/FashionMNIST/raw", train=False)  # ваш код
    plot_samples(train_data, labels_map, cols=3, rows=3)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().reshape(28, 28)
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    model.fc[0]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer, writer):
        size = len(dataloader.dataset)
        model.train()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute prediction error
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            writer.add_scalar("Loss/train", loss)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), (batch_idx + 1) * len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            ''' Чтобы сделать задание со сравнением лоссов, не забудьте 
                реализовать трекинг минимального лосса  '''

    writer = SummaryWriter()

    def test(dataloader: DataLoader,
             model: nn.Module,
             loss_fn: nn.Module,
             checkpoint_saver: CheckpointSaver,
             writer: SummaryWriter,
             step: int = 0) -> int:

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, num_correct = 0, 0
        precision = Precision(num_classes=10)
        recall = Recall(num_classes=10)
        with torch.no_grad():
            for inputs, targets in dataloader:
                step += 1
                preds = model(inputs)
                loss = loss_fn(preds, targets).item()
                writer.add_scalar("Loss/test", loss)
                num_correct += (preds.argmax(1) ==
                                targets).type(torch.float).sum().item()
                precision(preds, targets)
                recall(preds, targets)

        test_loss /= num_batches
        num_correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*num_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(
            f"Test Error: \n Precision: {(100*precision.compute()):>0.1f}%, Recall: {(100*recall.compute()):>0.1f} \n")
        writer.add_scalar("Loss/precision", precision.compute())
        writer.add_scalar("Loss/recall", recall.compute())
        checkpoint_saver.get_checkpoint(model, precision.compute(), step)
        return step

    checkpoint_saver = CheckpointSaver(
        model.__class__.__name__, should_minimize=False)
    step = 0

    for t in range(cfg.settings.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Train => ")
        train(train_dataloader, model, loss_fn, optimizer, writer)
        print("Test => ")
        step = test(test_dataloader, model, loss_fn,
                    checkpoint_saver, writer, step)

    torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    app()
