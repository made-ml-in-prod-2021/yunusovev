from ml_classifier.data.dataset import (
    read_data,
    save_data,
    split_data
)


def test_read_save_data(tmpdir, train_dataset, test_params):
    path = tmpdir.join(test_params.input_data_path)
    save_data(train_dataset, path)
    df_dump = read_data(path)
    assert train_dataset.equals(df_dump)


def test_split_data(train_dataset, test_params):
    train, val = split_data(train_dataset, test_params.splitting_params)
    assert train.shape[0] == int(test_params.train_size * (1 - test_params.splitting_params.val_size))
    assert val.shape[0] == int(test_params.train_size * test_params.splitting_params.val_size)
