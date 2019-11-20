import pandas as pd
import numpy as np
import operator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from eai_graph_tools.airflow_tasks import NODE_ABSENT, BENIGN, MALICIOUS, get_result_from_prediction_and_ground_truth
from xgboost import XGBClassifier
from eai_graph_tools.airflow_tasks.metrics_tasks import generate_metrics
from math import sqrt


def timestamp_convert(ts, print_details=False):
    """
        ts: either a unix time (int) or a datetime (str) such as "2015-01-22 11:50:14"
            returns the opposite
    """
    if isinstance(ts, int):
        pd_ts = pd.Timestamp(ts, unit='s')
        if print_details:
            print(f"{pd_ts} == {ts}")

    elif isinstance(ts, str):
        pd_ts = pd.Timestamp(ts).value // 10**9
        if print_details:
            print(f"{ts} == {pd_ts}")
    return pd_ts


def weight(error_k: float, i_k: np.array, k: np.array):
    # rrse
    r_k = sqrt(error_k / np.sum(np.square(i_k - np.mean(k))))
    w_k = 1 - min(1, r_k)
    return w_k


def get_model(model_type,
              max_depth=2,
              random_state=0,
              n_estimators=100):
    model = None
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)
    elif model_type == "MLP":
        assert "MLP model no yet supported"
    else:
        assert "Model unsupported"

    return model


def run_also(df,
             train_start,
             train_end,
             test_start,
             test_end,
             random_forest_max_depth,
             random_forest_random_state,
             random_forest_n_estimators,
             folds,
             samples_training_ratio,
             model_type="RandomForestRegressor"):

    assert folds == 1, "Also currently only supports folds=1."

    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df_train = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
    df_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
    df.drop(columns=['timestamp'], inplace=True)
    n_features = len(df.columns)
    n_testing_rows = len(df_test.index)
    scores = np.zeros((n_testing_rows, n_features))
    weights = np.zeros(n_features)

    for k_i, k in enumerate(df.columns):
        error_k = 0

        for f in range(folds):
            m_k = get_model(model_type,
                            max_depth=random_forest_max_depth,
                            random_state=random_forest_random_state,
                            n_estimators=random_forest_n_estimators)

            x_train = df_train.loc[:, df_train.columns != k]
            y_train = df_train.loc[:, [k]]

            x_test = df_test.loc[:, df_test.columns != k]
            y_test = df_test.loc[:, [k]]

            m_k.fit(x_train, y_train[k])
            pred_k = m_k.predict(x_test)

            print("pred_k: ", pred_k)
            print("np.square(y_test[k].values - pred_k)", np.square(y_test[k].values - pred_k))

            k_score = np.square(y_test[k].values - pred_k)
            k_score = np.transpose(k_score)

            scores[:, k_i] = k_score
            print("k_score: ", k_score)
            print("scores: ", scores)

            error_k = error_k + np.sum(k_score)

        print(f'Attribute k="{k}", error_k={error_k}')
        i_k = df.loc[:, df.columns == k]
        w_k = weight(error_k, i_k.values, y_test[k].values)
        print(f'w_k={w_k}\n\n')
        weights[k_i] = w_k

    predicted_scores = np.sqrt((1 / np.sum(weights) * np.sum(weights * scores, axis=1)))
    return predicted_scores


def load_ground_truth(raw_dataframe_filename,
                      train_start,
                      train_end,
                      test_start,
                      test_end):

    df = pd.read_hdf(raw_dataframe_filename, mode='r')
    df.loc[df.Label == 'MALICIOUS', 'Label'] = -1
    df.loc[df.Label == 'BENIGN', 'Label'] = 1

    y_train = df.Label[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
    y_test = df.Label[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
    df_inf = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]

    return df_inf, y_train, y_test


def load_features(features_dataframe_filename,
                  train_start,
                  train_end,
                  test_start,
                  test_end,
                  cont_variables=['all']):
    df = pd.read_hdf(features_dataframe_filename, mode='r')

    x_train = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
    x_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]

    if cont_variables[0] != 'all':
        x_train = x_train.loc[:, cont_variables]
        x_test = x_test.loc[:, cont_variables]
    else:
        x_train.drop(columns=['timestamp'], inplace=True)
        x_test.drop(columns=['timestamp'], inplace=True)

    return x_train, x_test


def create_grid_df(df,
                   start,
                   stop,
                   interval_width,
                   node_types_to_analyse=None):
    """
        Returns:
            - Pandas dataframe for visualizing metrics into a grid:
                | interval_id  | start_timestamp | end_timestamp | ip | pred_label | truth_label | \
                result (TP=0, TN=1, FP=2, FN=3, Node_absent=4)
    """
    df = df.sort_values(by=['timestamp'])

    # TODO: add node_types_to_analyse and make this generic...
    # node_types_to_analyse
    nodes = np.concatenate((df['Source IP'].unique(), df['Destination IP'].unique()), axis=0)
    nodes = np.unique(nodes)

    df_grid = pd.DataFrame(columns=["interval_id",
                                    "from",
                                    "to",
                                    "ip",
                                    "pred_label",
                                    "truth_label",
                                    "result"])

    assert stop >= (start + interval_width), "start, stop, interval_width parameters should be set " \
                                             "to get at least a single interval."
    # Split by intervals
    cnt = 0
    for i in range(start, stop, interval_width):
        if i + interval_width >= df['timestamp'].iloc[-1]:
            break

        df_interval = df.loc[(df['timestamp'] >= i) & (df['timestamp'] < (i + interval_width))]

        cnt = cnt + 1
        for n in nodes:
            result = NODE_ABSENT
            df_src = df_interval[df_interval['Source IP'] == n]
            df_dst = df_interval[df_interval['Destination IP'] == n]
            df_ = pd.concat([df_src, df_dst], axis=0)

            truth_label = 'n/a'
            pred_label = 'n/a'

            if len(df_.index) != 0:
                # Node found in interval
                truth_label = MALICIOUS if MALICIOUS in df_.truth_label.unique() else BENIGN
                pred_label = MALICIOUS if MALICIOUS in df_.pred_label.unique() else BENIGN
                result = get_result_from_prediction_and_ground_truth(pred_label, truth_label)

            df_grid = df_grid.append({'interval_id': cnt,
                                      'from': i,
                                      'to': i + interval_width,
                                      'ip': n,
                                      'pred_label': pred_label,
                                      'truth_label': truth_label,
                                      'result': result,
                                      }, ignore_index=True)

    return df_grid, nodes


def generate_interval_predictions(dataframe_filename,
                                  df_result,
                                  start,
                                  stop,
                                  interval_width):

    df, nodes = create_grid_df(df_result,
                               start,
                               stop,
                               interval_width)

    df.to_hdf(dataframe_filename, key='df', mode='w')


def generate_rowbased_predictions(dataframe_filename,
                                  df_inference,
                                  y_pred):

    df_inference['pred_label'] = y_pred
    df_inference['truth_label'] = df_inference.Label
    df_inference.to_hdf(dataframe_filename, key='df', mode='w')
    return df_inference


def fit_predict_local_outlier_factor(log,
                                     in_files,
                                     out_files,
                                     train_start,
                                     train_end,
                                     test_start,
                                     test_end,
                                     interval_width,
                                     cont_variables,
                                     cat_variables,
                                     n_neighbors,
                                     contamination,
                                     use_smote=False,
                                     smote_random_state=42):
    log.info("local_outlier_factor_fit_predict")
    # assert len(cat_variables) == 0, "Cat variables unsupported yet!"

    df_inference, y_train, y_test = load_ground_truth(in_files['raw_file'].path,
                                                      train_start,
                                                      train_end,
                                                      test_start,
                                                      test_end)

    x_train, x_test = load_features(in_files['features_file'].path,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                    cont_variables)

    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred_unsupervised_lof = clf.fit_predict(x_test)

    generate_metrics(y_test,
                     y_pred_unsupervised_lof,
                     out_files['row_based_metrics'].path)

    if use_smote:
        log.info("SMOTE not supported yet")
    else:
        df_result = generate_rowbased_predictions(out_files['df_all_predictions'].path,
                                                  df_inference,
                                                  y_pred_unsupervised_lof)

        generate_interval_predictions(out_files['df_interval_predictions'].path,
                                      df_result,
                                      test_start,
                                      test_end,
                                      interval_width)

    return "succeeded"


def fit_predict_random_forest_classifier(log,
                                         in_files,
                                         out_files,
                                         train_start,
                                         train_end,
                                         test_start,
                                         test_end,
                                         interval_width,
                                         cont_variables,
                                         cat_variables,
                                         n_estimators,
                                         max_depth,
                                         random_state):
    log.info("random_forest_classifier_fit_predict")
    assert len(cat_variables) == 0, "Cat variables unsupported yet!"

    df_inference, y_train, y_test = load_ground_truth(in_files['raw_file'].path,
                                                      train_start,
                                                      train_end,
                                                      test_start,
                                                      test_end)

    x_train, x_test = load_features(in_files['features_file'].path,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end,
                                    cont_variables)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    df_result = generate_rowbased_predictions(out_files['df_all_predictions'].path,
                                              df_inference,
                                              y_pred)

    generate_interval_predictions(out_files['df_interval_predictions'].path,
                                  df_result,
                                  test_start,
                                  test_end,
                                  interval_width)

    return "succeeded"


def fit_predict_random_forest_also(log,
                                   in_files,
                                   out_files,
                                   train_start,
                                   train_end,
                                   test_start,
                                   test_end,
                                   interval_width,
                                   cont_variables,
                                   mean_scaling_threshold=1.4,
                                   random_forest_max_depth=2,
                                   random_forest_random_state=0,
                                   random_forest_n_estimators=100,
                                   folds=1,
                                   samples_training_ratio=0.8,
                                   model_type='default'):
    log.info("fit_predict_random_forest_ALSO")
    print("fit_predict_random_forest_ALSO")
    df_inference, _, y_test = load_ground_truth(in_files['raw_file'].path,
                                                train_start,
                                                train_end,
                                                test_start,
                                                test_end)

    df_feat = pd.read_hdf(in_files['features_file'].path, mode='r')
    print("df_feat")

    predicted_scores = run_also(df_feat,
                                train_start=train_start,
                                train_end=train_end,
                                test_start=test_start,
                                test_end=test_end,
                                random_forest_max_depth=random_forest_max_depth,
                                random_forest_random_state=random_forest_random_state,
                                random_forest_n_estimators=random_forest_n_estimators,
                                folds=folds,
                                samples_training_ratio=samples_training_ratio,
                                model_type=model_type)
    print("run_also done...")

    # Convert prediction scores to -1/1 labels
    threshold_value = mean_scaling_threshold * np.mean(predicted_scores)
    y_pred_unsupervised_also = [-1 if score > threshold_value else 1
                                for score in predicted_scores]

    df_inference = df_inference.tail(len(predicted_scores))
    y_inference = y_test.tail(len(predicted_scores))
    df_predicted_scores = pd.DataFrame({"also_predicted_score": predicted_scores,
                                        "truth_label": y_inference,
                                        "pred_label": y_pred_unsupervised_also})
    df_predicted_scores.to_hdf(out_files['df_also_predicted_scores'].path, key='df', mode='w')

    generate_metrics(y_inference,
                     y_pred_unsupervised_also,
                     out_files['row_based_metrics'].path)

    df_result = generate_rowbased_predictions(out_files['df_all_predictions'].path,
                                              df_inference,
                                              y_pred_unsupervised_also)

    generate_interval_predictions(out_files['df_interval_predictions'].path,
                                  df_result,
                                  test_start,
                                  test_end,
                                  interval_width)

    return "succeeded"


def fit_predict_xgboost(log,
                        in_files,
                        out_files,
                        train_start,
                        train_end,
                        test_start,
                        test_end,
                        interval_width,
                        importance_type):
    log.info("xgboost_fit_predict")

    df_inference, y_train, y_test = load_ground_truth(in_files['raw_file'].path,
                                                      train_start,
                                                      train_end,
                                                      test_start,
                                                      test_end)

    x_train, x_test = load_features(in_files['features_file'].path,
                                    train_start,
                                    train_end,
                                    test_start,
                                    test_end)

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    feature_importance = model.get_booster().get_score(importance_type=importance_type)

    df_result = generate_rowbased_predictions(out_files['df_all_predictions'].path,
                                              df_inference,
                                              y_pred)

    generate_interval_predictions(out_files['df_interval_predictions'].path,
                                  df_result,
                                  test_start,
                                  test_end,
                                  interval_width)

    sorted_importance = sorted(feature_importance.items(), key=operator.itemgetter(1))
    with open(out_files['features_importance'].path, "w") as features_importance_file:
        features_importance_file.write(f"\n================xgboost_fit_predict================\n")
        features_importance_file.write(f"{sorted_importance}")

    return "succeeded"
