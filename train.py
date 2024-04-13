import mlflow
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import IteratorMode
from dataset.dataset import MazeData
from models.networks import U_Net
from utils.logger import get_logger

LOG = get_logger()
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def criterion(pred, label):
    return binary_cross_entropy(torch.sigmoid(pred), label)


def save_model(model, model_name) -> None:
    mlflow.pytorch.log_model(model, model_name)


def iterate(
    model: nn.Module,
    dataloader: DataLoader,
    mode: IteratorMode,
    device: torch.device,
    optimizer: torch.optim = None,
    prog_bar_prefix: str = "",
    prog_bar_update_rate: int = 10,
):
    if mode == IteratorMode.TRAIN and optimizer is None:
        raise Exception("Cannot train without optimizer")
    if mode == IteratorMode.TRAIN:
        prog_bar_pattern = " ="
        model.train()
    elif mode == IteratorMode.VALIDATE:
        prog_bar_pattern = " -"
        model.eval()
    else:
        prog_bar_pattern = " >"
        model.eval()

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
    pbar = tqdm(
        total=len(dataloader),
        bar_format=bar_format,
        desc=prog_bar_prefix,
        ascii=prog_bar_pattern,
    )

    torch.cuda.empty_cache()
    epoch_loss = 0
    context_dec = torch.enable_grad() if mode == IteratorMode.TRAIN else torch.no_grad()
    with context_dec:
        for i, (_, input, output) in enumerate(dataloader):

            input = input.to(device)
            output = output.to(device)

            if mode == IteratorMode.TRAIN:
                optimizer.zero_grad()

            pred = model(input)
            loss = criterion(pred, output)

            if mode == IteratorMode.TRAIN:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if i % prog_bar_update_rate == 0:
                pbar.update(min(prog_bar_update_rate, len(dataloader) - i))
                pbar.set_postfix(Loss=f"{loss.item():.4f}")

    epoch_loss /= len(dataloader)
    pbar.set_postfix(Loss=f"{epoch_loss:.4f}")
    pbar.close()

    return epoch_loss


def train(
    model: nn.Module,
    epochs: int,
    optimizer: torch.optim,
    train_dataloader: DataLoader,
    validate_dataloader: DataLoader,
    device: torch.device,
    save_rate: int = 50,
):
    best_train_loss = float("inf")
    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_epoch_loss = iterate(
            model,
            train_dataloader,
            IteratorMode.TRAIN,
            device,
            optimizer=optimizer,
            prog_bar_prefix=f"Epochs: {epoch + 1}/{epochs}",
        )
        mlflow.log_metric("train_loss", train_epoch_loss, epoch)

        if (epoch + 1) % save_rate == 0:
            save_model(model, f"epoch{epoch + 1:04d}")

        if train_epoch_loss < best_train_loss:
            LOG.info("Saving Model")
            best_train_loss = train_epoch_loss
            save_model(model, "best_train_model")

        val_loss = iterate(
            model,
            validate_dataloader,
            IteratorMode.VALIDATE,
            device,
            prog_bar_prefix="Validate",
        )
        mlflow.log_metric("val_loss", val_loss, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "best_val_model")

        LOG.info(f"Epoch {epoch + 1} complete")

    LOG.info("Training complete")


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    test_loss = iterate(
        model, dataloader, IteratorMode.TEST, device, prog_bar_prefix="Test"
    )
    print(f"Test dataset loss: {test_loss:.4f}")
    mlflow.log_metric("test_loss", test_loss)


def main():
    epochs = 500
    batch_size = 16
    num_workers = 4
    learning_rate = 1e-4
    model_path = None
    train_dataset_size = 1000
    validate_dataset_size = 200
    test_dataset_size = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U_Net(img_ch=1)

    if model_path is not None:
        model = model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = MazeData(train_dataset_size)
    valid_dataset = MazeData(validate_dataset_size)
    test_dataset = MazeData(test_dataset_size)

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=True, num_workers=num_workers
    )

    mlflow.log_params(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_fn": "binary_cross_entropy",
            "train_dataset_size": train_dataset_size,
            "validate_dataset_size": validate_dataset_size,
            "test_dataset_size": test_dataset_size,
        }
    )

    train(
        model,
        epochs,
        optimizer,
        train_dataloader,
        valid_dataloader,
        device,
        save_rate=50,
    )
    test(model, test_dataloader, device)


if __name__ == "__main__":
    main()
