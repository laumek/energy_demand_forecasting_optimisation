import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """LSTM model for sequence-to-vector energy demand forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.25):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
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
    def __init__(self, data, target, seq_length):
        self.data = data
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.target[index + self.seq_length - 1]
        return x, y


def prepare_lstm_data(X_train, X_test, y_train, y_test, seq_length, batch_size, test_batch_size=2048):
    """
    Scale pre-split train/test data and wrap it into DataLoaders for the LSTM.
    Expects X_train/X_test as DataFrames or arrays, y_train/y_test as 1D arrays.

    The last seq_length - 1 training rows are prepended to the test set before
    windowing, so the i-th test prediction targets y_test[i] and the LSTM covers
    the same timestamps as the non-sequence models.

    The model is stateless (every window starts from a zero hidden state), so the
    test set can be predicted in large batches; test_batch_size only affects speed.
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_dataset = TimeSeriesDataset(X_test_tensors, y_test_tensors, seq_length)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, x_scaler, y_scaler, y_train_values, y_test_values


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Train the LSTM, returning the trained model and per-epoch loss history.
    Stateless: each window starts from a zero hidden state, as at prediction time.
    """
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        losses = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs, _ = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss))

        epoch_losses.append(np.mean(losses))
        print(f"Epoch {epoch + 1}/{num_epochs}. Loss: {epoch_losses[-1]:.5f}.")

    return model, epoch_losses


def get_train_predictions(model, train_loader, y_train, y_scaler, step_size=336):
    """
    Return training-set predictions (inverse-scaled) plus mean/first-week/last-week
    MAE and RMSE. No plotting here — handled by the caller.
    """
    train_predictions_inverse = _predict(model, train_loader, y_scaler)

    # The first training window targets y_train[seq_length - 1]
    y_train_aligned = y_train[train_loader.dataset.seq_length - 1:]

    num_steps = len(train_predictions_inverse)
    mae_list, rmse_list = [], []

    for i in range(0, num_steps, step_size):
        end_idx = min(i + step_size, num_steps)
        actual_segment = y_train_aligned[i:end_idx]
        pred_segment = train_predictions_inverse[i:end_idx]

        mae_list.append(mean_absolute_error(actual_segment, pred_segment))
        rmse_list.append(np.sqrt(mean_squared_error(actual_segment, pred_segment)))

    return {
        "train_predictions": train_predictions_inverse,
        "mae_mean": np.mean(mae_list),
        "rmse_mean": np.mean(rmse_list),
        "mae_first_week": mae_list[0],
        "rmse_first_week": rmse_list[0],
        "mae_last_week": mae_list[-1],
        "rmse_last_week": rmse_list[-1],
        "mae_list": mae_list,
        "rmse_list": rmse_list,
    }


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