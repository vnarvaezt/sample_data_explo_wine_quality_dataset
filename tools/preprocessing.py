import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def ordinal_encoder(Xtrain, Xval, Xtest, features):
    if features is None:
        features = Xtrain.select_dtypes("category").columns.tolist()
    # fit encoder
    enc = OrdinalEncoder()
    _ = enc.fit(Xtrain[features])
    # infer model
    for df in [Xtrain, Xval, Xtest]:
        print(df.shape)
        df[features] = enc.transform(df[features])
        print(df.shape)
    return Xtrain, Xval, Xtest


# scale data


def scale_data(Xtrain, Xval, Xtest, features):
    if features is None:
        features = Xtrain.select_dtypes(np.number).columns.tolist()
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    for df in [Xtrain, Xval, Xtest]:
        df[features] = scaler.transform(df[features])
    return Xtrain, Xval, Xtest
