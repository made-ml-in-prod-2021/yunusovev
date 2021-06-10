from ml_classifier.features.preprocessing import (
    build_bin_pipeline,
    build_cat_pipeline,
    build_num_pipeline,
    build_column_selector,
    build_transformer
)


def test_build_cat_pipeline(train_dataset, val_dataset, test_params):
    cat_pipeline = build_cat_pipeline()
    tmp_train = train_dataset[test_params.feature_params.cat_features]
    tmp_val = val_dataset[test_params.feature_params.cat_features]
    cat_pipeline.fit(tmp_train)
    assert cat_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert cat_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert cat_pipeline.transform(tmp_train).shape[1] == cat_pipeline.transform(tmp_val).shape[1]


def test_build_num_pipeline(train_dataset, val_dataset, test_params):
    num_pipeline = build_num_pipeline()
    tmp_train = train_dataset[test_params.feature_params.num_features]
    tmp_val = val_dataset[test_params.feature_params.num_features]
    num_pipeline.fit(tmp_train)
    assert num_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert num_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert num_pipeline.transform(tmp_train).shape[1] == num_pipeline.transform(tmp_val).shape[1]


def test_build_bin_pipeline(train_dataset, val_dataset, test_params):
    bin_pipeline = build_bin_pipeline()
    tmp_train = train_dataset[test_params.feature_params.bin_features]
    tmp_val = val_dataset[test_params.feature_params.bin_features]
    bin_pipeline.fit(tmp_train)
    assert bin_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert bin_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert bin_pipeline.transform(tmp_train).shape[1] == bin_pipeline.transform(tmp_val).shape[1]


def test_columns_selector(train_dataset, val_dataset, test_params):
    column_selector = build_column_selector(test_params.feature_params)
    column_selector.fit(train_dataset)
    assert column_selector.transform(train_dataset).shape[0] == train_dataset.shape[0]
    assert column_selector.transform(val_dataset).shape[0] == val_dataset.shape[0]
    assert min(column_selector.transform(train_dataset).columns == column_selector.transform(val_dataset).columns)


def test_build_transformer(train_dataset, val_dataset, test_params):
    transformer = build_transformer(test_params.feature_params)
    transformer.fit(train_dataset)
    assert transformer.transform(train_dataset).shape[0] == train_dataset.shape[0]
    assert transformer.transform(val_dataset).shape[0] == val_dataset.shape[0]
    assert transformer.transform(train_dataset).shape[1] == transformer.transform(val_dataset).shape[1]
