import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np

from capsa import HistogramWrapper, MVEWrapper, EnsembleWrapper

data_path = "/data/hdmi_loans/Washington_State_HDMA-2016.csv"


def cat_encoder(df, exist_cols, new_cols):
    le = LabelEncoder()
    for exist, new in zip(exist_cols, new_cols):
        df.loc[:, new] = le.fit_transform(df.loc[:, exist])
    return df


def load_n_process_data_basic(normalize=False):
    df_main = pd.read_csv(data_path)

    df_main['loan_approved'] = df_main['action_taken_name'].apply(lambda x: 1 if x == 'Loan originated' else 0)

    non_nulls = ['loan_amount_000s', 'state_name', 'state_abbr', 'sequence_number', 'respondent_id',
                 'purchaser_type_name', 'property_type_name', 'preapproval_name', 'owner_occupancy_name',
                 'loan_type_name', 'loan_purpose_name', 'lien_status_name', 'hoepa_status_name',
                 'co_applicant_sex_name', 'co_applicant_race_name_1', 'co_applicant_ethnicity_name',
                 'as_of_year', 'application_date_indicator', 'applicant_sex_name', 'applicant_race_name_1',
                 'applicant_ethnicity_name', 'agency_name', 'agency_abbr', 'action_taken_name', 'loan_approved']

    # Find columns with null values
    null_cols = []
    sum_of_nulls = df_main.isna().sum()

    for idx in sum_of_nulls.index:
        if sum_of_nulls[idx] > 0:
            null_cols.append(idx)

    # print("Null cols : ", null_cols)

    # Drop the columns with null values
    df_preprocessed = df_main.drop(columns=null_cols)

    # print(df_preprocessed.info)

    # Convert cat to one hot, selecting all the cols with dtype == object
    cat_cols = df_preprocessed.select_dtypes("object").columns
    new_cat_cols_names = []

    # changing the name to cat, assumes old name ends in 'name'
    for old_name in cat_cols:
        new_name = old_name + '_cat'
        new_cat_cols_names.append(new_name)

    # print("cat_cols :", cat_cols)
    # print("new name :", new_cat_cols_names)

    df_preprocessed = cat_encoder(df_preprocessed, cat_cols, new_cat_cols_names)

    # Drop the old cat columns
    df_preprocessed = df_preprocessed.drop(columns=cat_cols)

    # encoded_features = []
    # for feature in new_cat_cols_names:
    #     encoded_feat = OneHotEncoder(categories='auto').fit_transform(
    #         df_preprocessed[feature].values.reshape(-1, 1)).toarray()
    #     n = df_preprocessed[feature].nunique()
    #     cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
    #     encoded_df = pd.DataFrame(encoded_feat, columns=cols)
    #     encoded_df.index = df_preprocessed.index
    #     encoded_features.append(encoded_df)
    #
    # # print(encoded_features)
    # # print("df_preproc: {}".format(df_preprocessed.columns))
    #
    # df_X_y_train_test = pd.concat([df_preprocessedcessed, *encoded_features], axis=1).drop(new_cat_cols_names, axis=1)
    df_X_y_train_test = df_preprocessed

    X_train, X_test, y_train, y_test = train_test_split(df_X_y_train_test.drop('loan_approved', axis=1),
                                                        df_X_y_train_test['loan_approved'], test_size=0.2,
                                                        random_state=42)

    # reset the indices
    for data in [X_train, X_test, y_train, y_test]:
        data.reset_index(drop=True, inplace=True)

    if normalize:
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scalerr.transform(X_test)

        # print current shapes
        # print('X_train shape = {} \n y_train shape = {}\n'.format(X_train_norm.shape, y_train.shape))
        # print('X_test shape = {} \n y_test shape = {}'.format(X_test_norm.shape, y_test.shape))

        return X_train_norm, y_train, X_test_norm, y_test
    else:
        print('X_train shape = {} \n y_train shape = {}\n'.format(X_train.shape, y_train.shape))
        print('X_test shape = {} \n y_test shape = {}'.format(X_test.shape, y_test.shape))
        return X_train, y_train, X_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_n_process_data_basic()
    model = keras.Sequential(
        [
            tf.keras.layers.Dense(50, activation="relu", name="layer1"),
            tf.keras.layers.Dense(10, activation="relu", name="layer2"),
            tf.keras.layers.Dense(1, activation="softmax", name="layer3"),
        ]
    )
    model.compile(optimizer="Adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))
    model.save("~/assets/hdmi")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



if __name__ == "__main__":
    main()