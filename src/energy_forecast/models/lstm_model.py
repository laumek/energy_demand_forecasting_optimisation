import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """LSTM model for sequence-to-vector energy demand forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.25):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch only applies dropout between stacked layers, so it has no effect with one layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        init.xavier_uniform_(self.fc.weight.data)
        self.fc.bias.data.fill_(0)

    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        else:
            h0, c0 = hidden

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)


class TimeSeriesDataset(Dataset):
    """
    Sliding windows of seq_length rows: window i covers rows i .. i + seq_length - 1 and
    targets row i + seq_length - 1. Indexing with a list of indices returns a whole batch
    in one gather, instead of building it one window at a time.
    """

    def __init__(self, data, target, seq_length):
        self.seq_length = seq_length
        # (n_windows, seq_length, n_features) view of data, no copy
        self.windows = data.unfold(0, seq_length, 1).permute(0, 2, 1)
        self.targets = target[seq_length - 1:]

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        return self.windows[index].contiguous(), self.targets[index]


def _make_loader(dataset, batch_size, shuffle=False, drop_last=False, seed=None):
    """DataLoader that fetches each batch with a single dataset[list_of_indices] call."""
    if shuffle:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size, drop_last), batch_size=None)


def prepare_lstm_data(
    X_train, X_test, y_train, y_test, seq_length, batch_size, test_batch_size=2048, shuffle=True, seed=42
):
    """
    Scale pre-split train/test data and wrap it into DataLoaders for the LSTM.
    Expects X_train/X_test as DataFrames or arrays, y_train/y_test as 1D arrays.

    The last seq_length - 1 training rows are prepended to the test set before
    windowing, so the i-th test prediction targets y_test[i] and the LSTM covers
    the same timestamps as the non-sequence models.

    The model is stateless (every window starts from a zero hidden state), so training
    windows are shuffled (seeded, for reproducibility) and the test set can be predicted
    in large batches; test_batch_size only affects speed.
    """
    X_train_values = X_train.values if hasattr(X_train, "values") else X_train
    X_test_values = X_test.values if hasattr(X_test, "values") else X_test
    y_train_values = np.asarray(y_train).reshape(-1, 1)
    y_test_values = np.asarray(y_test).reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_values)
    X_test_scaled = x_scaler.transform(X_test_values)

    y_train_scaled = y_scaler.fit_transform(y_train_values)
    y_test_scaled = y_scaler.transform(y_test_values)

    # Warm-up context from the end of the training set (already scaled with train-fitted scalers)
    context = seq_length - 1
    if context > 0:
        X_test_scaled = np.concatenate([X_train_scaled[-context:], X_test_scaled])
        y_test_scaled = np.concatenate([y_train_scaled[-context:], y_test_scaled])

    X_train_tensors = torch.from_numpy(X_train_scaled).float().to(DEVICE)
    y_train_tensors = torch.from_numpy(y_train_scaled).float().to(DEVICE)
    X_test_tensors = torch.from_numpy(X_test_scaled).float().to(DEVICE)
    y_test_tensors = torch.from_numpy(y_test_scaled).float().to(DEVICE)

    train_dataset = TimeSeriesDataset(X_train_tensors, y_train_tensors, seq_length)
    train_loader = _make_loader(train_dataset, batch_size, shuffle=shuffle, drop_last=True, seed=seed)

    test_dataset = TimeSeriesDataset(X_test_tensors, y_test_tensors, seq_length)
    test_loader = _make_loader(test_dataset, test_batch_size)

    return train_loader, test_loader, x_scaler, y_scaler, y_train_values, y_test_values


def _train_one_epoch(model, train_loader, criterion, optimizer):
    """One stateless training pass (each window starts from a zero hidden state); returns mean batch loss."""
    model.train()
    losses = []

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs, _ = model(batch_x)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def _loss_on(model, loader, criterion):
    """Mean loss (in scaled units) over every window in the loader."""
    model.eval()
    total, count = 0.0, 0

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            outputs, _ = model(batch_x.to(DEVICE))
            total += criterion(outputs, batch_y.to(DEVICE)).item() * len(batch_y)
            count += len(batch_y)

    return total / count


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Train the LSTM for a fixed number of epochs, returning the model and per-epoch loss history."""
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_losses.append(_train_one_epoch(model, train_loader, criterion, optimizer))
        print(f"Epoch {epoch + 1}/{num_epochs}. Loss: {epoch_losses[-1]:.5f}.")

    return model, epoch_losses


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs=30, patience=3):
    """
    Train until the validation loss hasn't improved for `patience` epochs, then restore the
    best weights. Returns (model, train_losses, val_losses, best_epoch), where best_epoch is
    the number of epochs that gave the lowest validation loss.
    """
    train_losses, val_losses = [], []
    best_val, best_epoch, best_state, epochs_without_improvement = np.inf, 0, None, 0

    for epoch in range(max_epochs):
        train_losses.append(_train_one_epoch(model, train_loader, criterion, optimizer))
        val_losses.append(_loss_on(model, val_loader, criterion))
        print(f"Epoch {epoch + 1}/{max_epochs}. Loss: {train_losses[-1]:.5f}, val loss: {val_losses[-1]:.5f}.")

        if val_losses[-1] < best_val:
            best_val, best_epoch, epochs_without_improvement = val_losses[-1], epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


def predict_train_set(model, train_loader, y_scaler, batch_size=2048):
    """
    Inverse-scaled predictions (flat array) for every training window, in time order and
    including the final partial batch that train_loader drops. The first window targets
    training row seq_length - 1, so prediction i corresponds to training row i + seq_length - 1.
    """
    loader = _make_loader(train_loader.dataset, batch_size)
    return _predict(model, loader, y_scaler).flatten()


def _predict(model, loader, y_scaler):
    """Stateless batched predictions over every window in the loader, inverse-scaled, shape (n, 1)."""
    model.eval()
    predictions = []

    with torch.inference_mode():
        for batch_x, _ in loader:
            preds, _ = model(batch_x.to(DEVICE))
            predictions.append(preds.cpu().numpy())

    return y_scaler.inverse_transform(np.concatenate(predictions, axis=0))


def predict_test_set(model, test_loader, y_test, y_scaler):
    """
    Predict every test timestamp and return inverse-scaled predictions (flat array) + metrics.
    The forecast horizon is set by the features (e.g. day-ahead lags), not by this function.
    """
    predictions = _predict(model, test_loader, y_scaler).flatten()
    y_test = np.asarray(y_test).flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {"predictions": predictions, "mae": mae, "rmse": rmse}
