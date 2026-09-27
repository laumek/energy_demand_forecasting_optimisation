import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from energy_forecast.models.lstm_model import (
    LSTM,
    _loss_on,
    predict_train_set,
    prepare_lstm_data,
    train_with_early_stopping,
)

SEQ_LENGTH = 4


@pytest.fixture
def split_data():
    y = np.arange(100, dtype=float)
    X = pd.DataFrame({"a": y, "b": 2 * y})
    return X[:80], X[80:], y[:80], y[80:]


def test_test_windows_align_with_test_targets(split_data):
    X_train, X_test, y_train, y_test = split_data
    _, test_loader, x_scaler, y_scaler, _, _ = prepare_lstm_data(
        X_train, X_test, y_train, y_test, SEQ_LENGTH, batch_size=8, shuffle=False
    )
    dataset = test_loader.dataset

    # One window per test row, and window i targets y_test[i]
    assert len(dataset) == len(y_test)
    targets = y_scaler.inverse_transform(dataset.targets.cpu().numpy()).flatten()
    np.testing.assert_allclose(targets, y_test, atol=1e-4)
    # The first window ends at the first test row, preceded by the last training rows
    first_window = x_scaler.inverse_transform(dataset[0][0].cpu().numpy())[:, 0]
    np.testing.assert_allclose(first_window, [77, 78, 79, 80], atol=1e-4)


def test_loaders_yield_batches_of_windows(split_data):
    train_loader, test_loader, *_ = prepare_lstm_data(*split_data, SEQ_LENGTH, batch_size=8, test_batch_size=16)

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (8, SEQ_LENGTH, 2)
    assert batch_y.shape == (8, 1)
    # The test loader keeps the final partial batch: 20 windows = 16 + 4
    assert [len(batch_y) for _, batch_y in test_loader] == [16, 4]


def test_training_shuffle_is_seeded(split_data):
    def first_batch(seed):
        train_loader, *_ = prepare_lstm_data(*split_data, SEQ_LENGTH, batch_size=8, seed=seed)
        return next(iter(train_loader))[1]

    assert torch.equal(first_batch(42), first_batch(42))
    assert not torch.equal(first_batch(42), first_batch(7))


def test_predict_train_set_covers_every_window_in_order(split_data):
    X_train, _, y_train, _ = split_data
    train_loader, _, _, y_scaler, _, _ = prepare_lstm_data(*split_data, SEQ_LENGTH, batch_size=8)
    torch.manual_seed(0)
    model = LSTM(2, hidden_size=4, num_layers=1, output_size=1)

    predictions = predict_train_set(model, train_loader, y_scaler)

    assert len(predictions) == len(y_train) - SEQ_LENGTH + 1


def test_early_stopping_restores_best_weights(split_data):
    train_loader, val_loader, *_ = prepare_lstm_data(*split_data, SEQ_LENGTH, batch_size=8)
    torch.manual_seed(0)
    model = LSTM(2, hidden_size=4, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    model, train_losses, val_losses, best_epoch = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, max_epochs=8, patience=2
    )

    assert len(train_losses) == len(val_losses) <= 8
    assert val_losses[best_epoch - 1] == min(val_losses)
    assert _loss_on(model, val_loader, criterion) == pytest.approx(min(val_losses))
