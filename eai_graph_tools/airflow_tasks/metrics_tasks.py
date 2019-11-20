import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
from sklearn.metrics import accuracy_score,\
    confusion_matrix,\
    balanced_accuracy_score,\
    classification_report, \
    f1_score
from eai_graph_tools.airflow_tasks import TP, TN, FP, FN, NODE_ABSENT, P, N


def generate_metrics(y_test,
                     predictions,
                     filename):

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    f1score = f1_score(y_test, predictions, average='weighted')

    print(f"Saving metrics to: {filename}")
    with open(filename, "w") as summary_file:
        summary_file.write(f"\n================Accuracy================\n")
        summary_file.write(f"Accuracy: {accuracy}\n")
        summary_file.write(f"Balanced Accuracy: {balanced_accuracy}\n")
        summary_file.write(f"\n=========Classification Report==========\n")
        summary_file.write(f"{class_report}\n")
        summary_file.write(f"\n===========Confusion Matrix=============\n")
        summary_file.write(f"{conf_matrix}\n")
        summary_file.write(f"\n=========F1 Score (weighted) ===========\n")
        summary_file.write(f"{f1score}\n")

    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Classification Report: {class_report}")
    print(f"Confusion Matrix{conf_matrix}")
    print(f"F1 Score: {f1score}")


def generate_interval_scores_from_samples_prediction(df, start, stop, step):
    """
        Returns:
            - Pandas dataframe for calculating metrics:
                | interval_id  | start_timestamp | end_timestamp | ip | pred_label | truth_label |
            - Pandas dataframe for visualizing metrics into a grid:
                | interval_id  | start_timestamp | end_timestamp | ip | pred_label |
                | truth_label  | result (TP=0, TN=1, FP=2, FN=3, Node_absent=4)
    """
    df = df.sort_values(by=['timestamp'])
    nodes = np.concatenate((df.scrip.unique(), df.dstip.unique()), axis=0)
    nodes = np.unique(nodes)

    df_res = pd.DataFrame(columns=["interval_id",
                                   "from",
                                   "to",
                                   "ip",
                                   "pred_label",
                                   "truth_label"])
    df_grid = pd.DataFrame(columns=["interval_id",
                                    "from",
                                    "to",
                                    "ip",
                                    "result"])

    # Split by intervals
    cnt = 0
    for i in range(start, stop, step):
        if i + step - 1 >= df.index[-1]:
            break
        df_interval = df.loc[i:i + step - 1]
        cnt = cnt + 1
        for n in nodes:
            result = NODE_ABSENT
            df_src = df_interval[df_interval.scrip == n]
            df_dst = df_interval[df_interval.dstip == n]
            df_ = pd.concat([df_src, df_dst], axis=0)

            if len(df_.index) != 0:
                # Node found in interval
                ground_truth_label = 0
                if 1 in df_.label.unique():
                    ground_truth_label = 1

                pred_label = 0
                if 1 in df_.pred_label.unique():
                    pred_label = 1

                if ground_truth_label == 1 and pred_label == 1:
                    result = TP
                elif ground_truth_label == 0 and pred_label == 0:
                    result = TN
                elif ground_truth_label == 1 and pred_label == 0:
                    result = FN
                elif ground_truth_label == 0 and pred_label == 1:
                    result = FP

                df_res = df_res.append({'interval_id': cnt,
                                        'from': i,
                                        'to': i + step - 1,
                                        'ip': n,
                                        'pred_label': pred_label,
                                        'truth_label': ground_truth_label}, ignore_index=True)

            df_grid = df_grid.append({'interval_id': cnt,
                                      'from': i,
                                      'to': i + step - 1,
                                      'ip': n,
                                      'result': result,
                                      }, ignore_index=True)

    return df_res, df_grid, nodes


def get_data_for_grid(df):
    nodes = np.array(df.ip.unique())
    intervals = df.interval_id.unique()
    np_data_grid = np.zeros((len(nodes), len(intervals)))

    for interval in intervals:
        np_data_grid[:, interval - 1] = df[(df.interval_id == interval)].result
    return np_data_grid, nodes, intervals


def plot_grid(data,
              intervals,
              nodes,
              filename,
              interval_width_s=None,
              title=None,
              x_axis_range=None,
              has_ground_truth=None):

    if x_axis_range is not None:
        data = data[:, x_axis_range[0]:x_axis_range[1]]

    missing_color = "white"
    # TP=0, TN=1, FP=2, FN=3, NODE_ABSENT=4, P=5, N=6
    if has_ground_truth:
        tp_color = 'lime'
        fn_color = 'olivedrab'
        tn_color = 'red'
        fp_color = 'rosybrown'
        cmap = colors.ListedColormap([tp_color, tn_color, fp_color, fn_color, missing_color])
    else:
        p_color = 'lime'
        n_color = 'red'
        np.where(data == P, 0, data)
        np.where(data == N, 1, data)
        np.where(data == NODE_ABSENT, 2, data)
        cmap = colors.ListedColormap([missing_color, p_color, n_color])

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(150, 25))
    ax.imshow(data, cmap=cmap, norm=norm)

    y_major_ticks = np.arange(0, len(nodes), 1)
    y_minor_ticks = np.arange(-.5, len(nodes), 1)

    if x_axis_range is not None:
        x_major_ticks = np.arange(x_axis_range[0], x_axis_range[1], 1)
        x_minor_ticks = np.arange(x_axis_range[0] - .5, x_axis_range[1], 1)
    else:
        x_major_ticks = np.arange(0, len(intervals), 1)
        x_minor_ticks = np.arange(-.5, len(intervals), 1)

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)

    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.set_yticklabels(labels=nodes)

    ax.grid(which='minor', alpha=1.0, color='black', linewidth=2)
    ax.grid(which='major', alpha=0.0)

    if has_ground_truth:
        tp_patch = mpatches.Patch(color=tp_color, label='TP')
        fn_patch = mpatches.Patch(color=fn_color, label='FN')
        tn_patch = mpatches.Patch(color=tn_color, label='TN')
        fp_patch = mpatches.Patch(color=fp_color, label='FP')
        missing_patch = mpatches.Patch(color=missing_color, label='Missing')
        plt.legend(handles=[tp_patch, fn_patch, tn_patch, fp_patch, missing_patch])
    else:
        p_patch = mpatches.Patch(color=p_color, label='P')
        n_patch = mpatches.Patch(color=n_color, label='N')
        missing_patch = mpatches.Patch(color=missing_color, label='Missing')
        plt.legend(handles=[p_patch, n_patch, missing_patch])

    plt.xlabel(f'Intervals (width= {interval_width_s if interval_width_s is not None else "-"} seconds)')
    plt.ylabel('Nodes')

    if title is not None:
        plt.title(title, fontsize='x-large')

    print(f"Saving grid view to: {filename}")
    plt.savefig(filename)


def create_interval_metrics(log,
                            in_files,
                            out_files,
                            interval_width,
                            title):
    log.info("create_interval_metrics")
    pred_df = pd.read_hdf(in_files['prediction_df'].path, 'df')
    np_data_grid, nodes, intervals = get_data_for_grid(pred_df)

    # Check if data is based on a ground truth (TP=0, TN=1, FP=2, FN=3, NODE_ABSENT=4)
    # or just contains predictions (P,N)
    unique_results = np.unique(np_data_grid)
    has_ground_truth = False if P in unique_results or N in unique_results else True

    plot_grid(np_data_grid,
              intervals,
              nodes,
              filename=out_files['grid_png'].path,
              interval_width_s=interval_width,
              title=title,
              has_ground_truth=has_ground_truth)

    if has_ground_truth:
        trimmed_pred_df = pred_df[pred_df.truth_label != 'n/a']
        log.info(f"Removed {len(pred_df.index) - len(trimmed_pred_df.index)} 'n/a' results "
                 f"for generating global metrics.")

        generate_metrics(np.array(trimmed_pred_df.truth_label, dtype=int),
                         np.array(trimmed_pred_df.pred_label, dtype=int),
                         filename=out_files['metrics_summary_file'].path)
    else:
        log.info("Skipping metrics calculation - experiment doesn't contain ground truth")
    return 'succeeded'
