import copy
import attr
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from airflow.models import Variable
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from eai_graph_tools.graph_tools.exporters.pandas.export_h5 import export_dataframe_to_h5
from eai_graph_tools.airflow_tasks.model_tasks import InferenceOutput
from eai_graph_tools.airflow_tasks import TP, TN, FP, FN, NODE_ABSENT, P, N
from eai_graph_tools.datasets.intervals import load_dataset_vendor
from eai_graph_tools.airflow_tasks import get_all_nodes_from_dataset, BENIGN, MALICIOUS, UNLABELLED
from eai_graph_tools.datasets import convert_label_to_numerical
from sklearn.metrics import auc, confusion_matrix, classification_report


def frange(start, stop, step):
    i = start
    while i < stop:
        yield round(i, 3)
        i += step


def plot_roc(noi, fpr, tpr, roc_auc, output_path, ref_embed_config=''):
    plt.figure()
    lw = 2  # ???
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for NODE:{noi}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, f"roc_{noi}_{ref_embed_config}.png"))
    plt.close()
    return


def multiplot_roc(noi, curves, output_path):
    plt.figure()
    lw = 2  # ???

    for curve in curves:
        plt.plot(curve['fpr'],
                 curve['tpr'],
                 color=curve['color'],
                 lw=lw,
                 label=f'ROC {curve["label"]} (area = {curve["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for NODE:{noi}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, f"roc_{noi}_all.png"))
    plt.close()
    return


def plot_node_analysis(node,
                       np_embeddings,
                       interval_count,
                       y_true_test,
                       y_pred_test,
                       output_path,
                       svm_nu,
                       ref_embed_config=''):
    signals = []

    x = range(0, interval_count)
    dims = np_embeddings.shape[1]
    for i in range(1, dims + 1):
        signal = {}
        signal['label'] = f"emb_{i}"
        signal['marker'] = ''
        signal['name'] = f'y{i}'
        signal['color'] = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        signal['x'] = x
        signal['y'] = np_embeddings[:, i - 1]
        signals.append(signal)

    signals.append({'name': f'y{dims + 1}',
                    'x': x,
                    'y': y_true_test,
                    'label': 'TRUTH',
                    'marker': 'o',
                    'color': 'green'})
    signals.append({'name': f'y{dims + 2}',
                    'x': x,
                    'y': y_pred_test,
                    'label': 'PRED',
                    'marker': '+',
                    'color': 'yellow'})

    plt.figure()
    plt.title(f'Embeddings Analysis Node:{node} svm_nu={svm_nu}')
    for signal in signals:
        plt.plot(signal['x'], signal['y'],
                 color=signal['color'],
                 linewidth=1,
                 label=signal['label'],
                 marker=signal['marker'])

    plt.legend()
    plt.savefig(os.path.join(output_path, f"embeddings_analysis_{node}_{ref_embed_config}_svm_nu_{svm_nu:.2f}.png"))
    plt.close()


def get_labeled_embeddings(df_metrics, node, start, end):
    """
    Returns the labelled embeddings of a node over multiple intervals (as separate list items)
    """
    df_embeddings_subset = df_metrics[(df_metrics.timestamp >= start.value) & (df_metrics.timestamp < end.value)]
    df_embeddings_subset = df_embeddings_subset.loc[df_embeddings_subset['node'] == node]

    labeled_data = {
        "embeddings": [],
        "class_label": [],
        "class_label_int": [],
    }

    for _, row in df_embeddings_subset.iterrows():
        labeled_data["class_label"].append(row['class_label'])
        if row['embeddings'] == "n/a":
            labeled_data["embeddings"].append("n/a")
            labeled_data["class_label_int"].append("n/a")
        else:
            labeled_data["embeddings"].append(np.fromstring(str(row['embeddings']).replace('[', '').replace(']', ''),
                                                            dtype=float,
                                                            sep=','))
            labeled_data["class_label_int"].append(convert_label_to_numerical(row['class_label']))

    return labeled_data, df_embeddings_subset


def create_metrics_df(node_embeddings,
                      inference_dataset,
                      nodes_of_interest,
                      training_intervals_count,
                      generate_cosine_only=True):

    baselines = {}
    for noi in tqdm(nodes_of_interest):
        training_intervals_limit = int(training_intervals_count) - 1
        baselines[noi] = np.nanmean(node_embeddings[noi][:training_intervals_limit], axis=0)

    df_embeddings = pd.DataFrame()

    for k, interval_data in tqdm(enumerate(inference_dataset)):
        interval_start = inference_dataset.start + k * inference_dataset.interval_width
        interval_end = inference_dataset.start + (k + 1) * inference_dataset.interval_width
        interval_info = interval_data.y  # Tensor with malicious (-1) and benign (1) info for the given interval,
        # 0 if unlabelled
        node_indexes_in_tensors = interval_data.node_indexes_in_tensors

        for noi in nodes_of_interest:
            training_intervals_limit = int(training_intervals_count) - 1

            if noi in node_indexes_in_tensors \
                    and not np.isnan(baselines[noi]).any() \
                    and not np.isnan(node_embeddings[noi][k]).any()\
                    and len(node_embeddings[noi][:training_intervals_limit]) != 0:

                label = interval_info[node_indexes_in_tensors[noi]].item() \
                    if interval_info[node_indexes_in_tensors[noi]].item() != 0 else 'n/a'

                if generate_cosine_only:
                    row_values = {
                        'node': noi,
                        'baseline_embeddings': np.array2string(baselines[noi], separator=', '),
                        'embeddings': np.array2string(node_embeddings[noi][k], separator=', '),
                        'class_label': label,
                        'timestamp': interval_start,     # interval_info['timestamp'],
                        'interval_duration': interval_end - interval_start,
                        'interval_idx': k,
                        'cosine': pairwise_distances([baselines[noi]],
                                                     [node_embeddings[noi][k]],
                                                     metric='cosine')[0][0],
                    }
                else:
                    row_values = {
                        'node': noi,
                        'baseline_embeddings': np.array2string(baselines[noi], separator=', '),
                        'embeddings': np.array2string(node_embeddings[noi][k], separator=', '),
                        'class_label': label,
                        'timestamp': interval_info['timestamp'],
                        'interval_duration': interval_end - interval_start,
                        'interval_idx': k,
                        'cosine': pairwise_distances([baselines[noi]],
                                                     [node_embeddings[noi][k]],
                                                     metric='cosine')[0][0],
                        'cityblock': pairwise_distances([baselines[noi]],
                                                        [node_embeddings[noi][k]],
                                                        metric='cityblock')[0][0],
                        'euclidean': pairwise_distances([baselines[noi]],
                                                        [node_embeddings[noi][k]],
                                                        metric='euclidean')[0][0],
                        'l1': pairwise_distances([baselines[noi]], [node_embeddings[noi][k]], metric='l1')[0][0],
                        'l2': pairwise_distances([baselines[noi]], [node_embeddings[noi][k]], metric='l2')[0][0],
                        'braycurtis': pairwise_distances([baselines[noi]],
                                                         [node_embeddings[noi][k]],
                                                         metric='braycurtis')[0][0],
                        'canberra': pairwise_distances([baselines[noi]],
                                                       [node_embeddings[noi][k]],
                                                       metric='canberra')[0][0],
                        'chebyshev': pairwise_distances([baselines[noi]],
                                                        [node_embeddings[noi][k]],
                                                        metric='chebyshev')[0][0],
                        'correlation': pairwise_distances([baselines[noi]],
                                                          [node_embeddings[noi][k]],
                                                          metric='correlation')[0][0],
                        'hamming': pairwise_distances([baselines[noi]],
                                                      [node_embeddings[noi][k]],
                                                      metric='hamming')[0][0],
                        'minkowski': pairwise_distances([baselines[noi]],
                                                        [node_embeddings[noi][k]],
                                                        metric='minkowski')[0][0],
                        'sqeuclidean': pairwise_distances([baselines[noi]],
                                                          [node_embeddings[noi][k]],
                                                          metric='sqeuclidean')[0][0],
                    }
            else:
                row_values = {
                    'node': noi,
                    'baseline_embeddings': 'n/a',
                    'embeddings': 'n/a',
                    'class_label': 'n/a',
                    'timestamp': interval_start,  # interval_info['timestamp'],
                    'interval_duration': interval_end - interval_start,
                    'interval_idx': k,
                    'cosine': 'n/a',
                }
            df_embeddings = df_embeddings.append(row_values, ignore_index=True)

    df_embeddings.timestamp = df_embeddings.timestamp.astype(int)
    df_embeddings.interval_idx = df_embeddings.interval_idx.astype(int)
    return df_embeddings


@attr.s(auto_attribs=True)
class CreatePostProcessingDFOutput():
    df1_detailed_classifier_data: pd.DataFrame = attr.ib(default=None)
    df2_roc_classifier_data: pd.DataFrame = attr.ib(default=None)
    df_metrics: pd.DataFrame = attr.ib(default=None)
    output_path: os.PathLike = attr.ib(default=".")
    experiment_name: str = attr.ib(default="0")

    def save(self):
        output_file = os.path.join(self.output_path, f'analysis_{self.experiment_name}.h5')
        self.df1_detailed_classifier_data.to_hdf(output_file, key='df1_detailed_classifier_data', mode='w')
        self.df2_roc_classifier_data.to_hdf(output_file, key='df2_roc_classifier_data', mode='w')
        self.df_metrics.to_hdf(output_file, key='df', mode='w')

        # Saving df standalone as well to make it easier to visualize with dfgui
        output_file = os.path.join(self.output_path, f'df1_detailed_classifier_data.h5')
        export_dataframe_to_h5(self.df1_detailed_classifier_data, output_file)
        output_file = os.path.join(self.output_path, f'df2_roc_classifier_data.h5')
        export_dataframe_to_h5(self.df2_roc_classifier_data, output_file)


def prepare_embeddings_ref_node(df, ref_node, start, end, output_path):
    embeds_ref_node, _ = get_labeled_embeddings(df, ref_node, start, end)
    print(f"Get labeled embeddings for ref_node: {ref_node}")
    print(f"embeds_ref_node: {embeds_ref_node}")

    file_path_ref_embeds_ref_node = os.path.join(output_path, f"ref_embeds_node_{ref_node}.p")
    os.makedirs(output_path, exist_ok=True)
    pickle.dump(embeds_ref_node['embeddings'], open(file_path_ref_embeds_ref_node, "wb"))
    return embeds_ref_node


def prepare_embeddings_ref_node_list(df, ref_node_list, start, end, output_path):
    print(f'Reference Nodes: {ref_node_list}')

    embeds_ref_node_list = []
    for node in ref_node_list:
        labeled_data, _ = get_labeled_embeddings(df, node, start, end)
        embeds_ref_node_list.extend(labeled_data['embeddings'])
    file_path_ref_nodes_embeds = os.path.join(output_path, "ref_node_list_embeds.p")
    pickle.dump(embeds_ref_node_list, open(file_path_ref_nodes_embeds, "wb"))
    return embeds_ref_node_list


def predict(log,
            in_files,
            out_files,
            start,
            end,
            interval_width,
            svm_training_technique,
            nodes_of_interest,
            reference_nodes,
            reference_victim_node,
            airflow_vars):

    log.info("predict")
    print(f"svm_training_technique: {svm_training_technique}")
    svm_kernel = 'rbf'
    svm_gamma = 'scale'
    svm_nu = 0.15

    output_path = os.path.dirname(out_files['df_metrics'].path)
    print(f"Predict saving to {output_path}")

    inf_output = InferenceOutput(filename=in_files['node_embeddings'].path)
    inf_output.restore()

    dataset_input_files = in_files
    del dataset_input_files['node_embeddings']
    del dataset_input_files['trained_model']

    inference_dataset = load_dataset_vendor(dataset_input_files,
                                            start,
                                            interval_width)

    training_intervals_count = Variable.get(airflow_vars['training_intervals_count'], default_var=0)

    if len(nodes_of_interest) == 0:
        log.warning("Node list not specified: running prediction on entire node list (might be slow!!)")
        nodes_of_interest = get_all_nodes_from_dataset(inference_dataset)

    # Generating metrics for selected nodes
    df = create_metrics_df(inf_output.node_embeddings,
                           inference_dataset,
                           nodes_of_interest,
                           training_intervals_count,
                           generate_cosine_only=True)

    df['index'] = df['timestamp']
    df.set_index('index', inplace=True)
    df.to_hdf(out_files['df_metrics'].path, key='df', mode='w')

    print(f"start: {start}")
    print(f"end: {end}")

    embeds_ref_node = prepare_embeddings_ref_node(df, reference_victim_node, start, end, output_path)
    embeds_ref_node_list = prepare_embeddings_ref_node_list(df, reference_nodes, start, end, output_path)

    node_truth = {}
    node_predictions = {}

    interval_len = 0

    for node in nodes_of_interest:
        print(f"processing node: {node}")
        # Get test set's embeddings
        x_testing_embeddings, _ = get_labeled_embeddings(df, node, start, end)
        x_test = x_testing_embeddings["embeddings"]

        # Reuse y tensor here?
        y_true_test = x_testing_embeddings["class_label_int"]  # 1 for inliers, -1 for outliers, 0 for unlabelled

        print(f"interval count: {len(y_true_test)}")
        assert interval_len == 0 or len(y_true_test) == interval_len, "Inference length mismatch!"
        interval_len = len(y_true_test)
        node_truth[node] = y_true_test

        # Determine the node label for the entire interval
        node_global_label = 'malicious' if len(set(y_true_test)) > 1 else 'benign'
        # len > 1 means the set contains more than 1 unique value, thus benign and malicious exist.

        # Generate a single reference embeddings
        if svm_training_technique == 'self':
            x_ref = x_test  # 'ref1'
        elif svm_training_technique == 'self_plus_victim':
            # 2- Embeddings from the node itself + a victim node
            x_ref = copy.deepcopy(embeds_ref_node['embeddings'])
            x_ref.extend(x_test)
        elif svm_training_technique == 'self_plus_reference_nodes':
            # 3- Embeddings from the node + all_victims + a few non_victims
            x_ref = copy.deepcopy(embeds_ref_node_list)
        else:
            raise ValueError(f'invalid svm_training_technique: "{svm_training_technique}", supported values: "self", '
                             '"self plus victim", "self plus reference nodes"')

        # Need to drop N/A intervals here (either from x_ref or x_test)
        x_ref_incomplete_idx = []
        for i, j in enumerate(x_ref):
            if type(j) == str:          # str for "n/a", array otherwise
                x_ref_incomplete_idx.append(i)

        x_test_incomplete_idx = []
        for i, j in enumerate(x_test):
            if type(j) == str:
                x_test_incomplete_idx.append(i)

        # 2 is to allow a minimum of 2 intervals for training the OC-SVM (add as a config param?)
        if len(x_ref) >= (len(x_ref_incomplete_idx) + 2) and len(x_test) > len(x_test_incomplete_idx):
            if len(x_ref_incomplete_idx) > 0:
                print(f"Dropping intervals because of incomplete reference data for node: {node}")
                print(f"\tIntervals: {x_ref_incomplete_idx}")
            if len(x_test_incomplete_idx) > 0:
                print(f"Dropping intervals because of incomplete embeddings for node: {node}")
                print(f"\tIntervals: {x_test_incomplete_idx}")

            # starting from the end, removing items from list
            for i in reversed(x_ref_incomplete_idx):
                del(x_ref[i])
            for i in reversed(x_test_incomplete_idx):
                del(x_test[i])

            try:
                print("Fitting OC-SVM")
                clf = svm.OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma)
                clf.fit(x_ref)
                y_pred_test = clf.predict(x_test)   # 1 for inliers, -1 for outliers.
                node_predictions[node] = y_pred_test

                # Put back missing intervals at the right place.
                for i in x_test_incomplete_idx:
                    if i < node_predictions[node].size:
                        node_predictions[node] = np.insert(node_predictions[node], i, [NODE_ABSENT])
                    else:
                        node_predictions[node] = np.append(node_predictions[node], [NODE_ABSENT])

            except Exception as e:
                assert False, e
        else:
            print(f"All intervals dropped for node: {node}")
            node_predictions[node] = [NODE_ABSENT] * interval_len

        print(f"---> Node:{node}\t Node global label: {node_global_label}")

    predictions = []

    for k, interval_data in tqdm(enumerate(inference_dataset)):
        interval_start = inference_dataset.start + k * inference_dataset.interval_width
        interval_end = inference_dataset.start + (k + 1) * inference_dataset.interval_width

        print(f"Getting results for interval: {k}")
        for node in nodes_of_interest:
            print(f"Node: {node}")

            pred_label = node_predictions[node][k]
            truth_label = node_truth[node][k]

            if (truth_label == UNLABELLED or truth_label == 'n/a') and pred_label == MALICIOUS:
                result = P
            elif (truth_label == UNLABELLED or truth_label == 'n/a') and pred_label == BENIGN:
                result = N
            elif truth_label == MALICIOUS and pred_label == MALICIOUS:
                result = TP
            elif truth_label == BENIGN and pred_label == BENIGN:
                result = TN
            elif truth_label == MALICIOUS and pred_label == BENIGN:
                result = FN
            elif truth_label == BENIGN and pred_label == MALICIOUS:
                result = FP
            else:
                result = NODE_ABSENT

            pred = {'interval_id': k,
                    'start_timestamp': interval_start,
                    'end_timestamp': interval_end,
                    'ip': node,
                    'pred_label': pred_label,
                    'truth_label': truth_label,
                    'result': result}
            predictions.append(pred)
            print(f"Appending to predictions: {pred}")

    print("Saving prediction dataframe - DONE")
    prediction_df = pd.DataFrame(predictions)

    # Pandas dataframe  visualizing metrics into a grid:
    #     | interval_id | start_timestamp | end_timestamp | ip | pred_label | truth_label | /
    #     | result (TP=0, TN=1, FP=2, FN=3, Node_absent=4, POSITIVE=5, NEGATIVE=6) |
    #
    prediction_df.to_hdf(out_files['prediction_df'].path, key='df', mode='w')

    return 'succeeded'


def nodes_analysis(log,
                   in_files,
                   out_files,
                   experiment_name,
                   start,
                   end,
                   nodes_of_interest,
                   reference_nodes,
                   reference_victim_node
                   ):

    log.info("Node Analysis")
    svm_kernel = 'rbf'
    svm_gamma = 'scale'
    output_path = os.path.dirname(out_files['metrics_summary_file'].path)

    print(f"Nodes analysis, saving to {output_path}")
    with open(out_files['metrics_summary_file'].path, "w") as summary_file:

        summary_file.write(f"\nMetrics data {experiment_name}\n\n")
        df = pd.read_hdf(in_files['df_metrics'].path, key='df')

        embeds_ref_node = prepare_embeddings_ref_node(df, reference_victim_node, start, end, output_path)
        embeds_ref_node_list = prepare_embeddings_ref_node_list(df, reference_nodes, start, end, output_path)

        target_names = ['class MALICIOUS', 'class BENIGN']  # Confirm class ordering
        target_names_report = ['class MALICIOUS', 'class BENIGN', 'weighted avg']  # Confirm class ordering

        nu_values = []
        roc_values = []
        for node in nodes_of_interest:

            # Get test set's embeddings
            x_testing_embeddings, _ = get_labeled_embeddings(df, node, start, end)
            x_test = x_testing_embeddings["embeddings"]
            y_true_test = x_testing_embeddings["class_label_int"]

            # Determine the node label for the entire interval
            node_global_label = 'malicious' if MALICIOUS in y_true_test else 'benign'

            # len > 1 means the set contains more than 1 unique value, thus bening and malicious exist.
            summary_file.write(f"NODE: {node}\t Global Label: {node_global_label}\n")

            # Generate a few reference embeddings
            reference_embeddings_configs = []
            # 1- Embeddings from the node itself only (works well on victim nodes)
            ref_embeds_self = x_test
            reference_embeddings_configs.append(('self', ref_embeds_self))     # itself only

            # 2- Embeddings from the node itself + a victim node
            ref_embeds_self_plus_victim = copy.deepcopy(embeds_ref_node['embeddings'])
            ref_embeds_self_plus_victim.extend(x_test)
            reference_embeddings_configs.append(('self_plus_victim', ref_embeds_self_plus_victim))

            # 3- Embeddings from the node + all_victims + a few non_victims
            self_plus_reference_nodes = copy.deepcopy(embeds_ref_node_list)
            reference_embeddings_configs.append(('self_plus_reference_nodes', self_plus_reference_nodes))

            for ref_embeds_cfg in reference_embeddings_configs:
                fpr = []
                tpr = []
                nu = []

                skip_roc_plot = False

                try:
                    for svm_nu in frange(0.05, 1.0, 0.05):
                        nu.append(svm_nu)
                        clf = svm.OneClassSVM(nu=svm_nu, kernel=svm_kernel, gamma=svm_gamma)

                        x_ref = ref_embeds_cfg[1]
                        clf.fit(x_ref)

                        y_pred_test = clf.predict(x_test)

                        plot_node_analysis(node=node,
                                           np_embeddings=np.array(x_test),
                                           interval_count=len(x_test),
                                           y_true_test=y_true_test,
                                           y_pred_test=y_pred_test,
                                           output_path=output_path,
                                           svm_nu=svm_nu,
                                           ref_embed_config=ref_embeds_cfg[0])

                        tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred_test).ravel()
                        tn = float(tn)
                        fp = float(fp)
                        fn = float(fn)
                        tp = float(tp)

                        fpr_sample = fp / (fp + tn) if fp != 0 else 0.0
                        tpr_sample = tp / (tp + fn) if tp != 0 else 0.0

                        fpr.append(fpr_sample)
                        tpr.append(tpr_sample)

                        report_txt = classification_report(y_true_test,
                                                           y_pred_test,
                                                           target_names=target_names,
                                                           output_dict=False)

                        if svm_nu == 0.15:
                            summary_file.write(f"=== svm_nu: {svm_nu}  SVM training setup:{ref_embeds_cfg[0]}===\n")
                            summary_file.write(f"classification_report:\n{report_txt}\n")

                        report = classification_report(y_true_test,
                                                       y_pred_test,
                                                       target_names=target_names,
                                                       output_dict=True)

                        for target_name in target_names_report:
                            nu_values.append({'node': node,
                                              'node_global_label': node_global_label,
                                              'classifier': 'OneClassSVM',
                                              'hyperparam1': ref_embeds_cfg[0],
                                              'hyperparam2': svm_nu,
                                              'target': target_name,
                                              'precision': report[target_name]['precision'],
                                              'recall': report[target_name]['recall'],
                                              'f1-score': report[target_name]['f1-score'],
                                              'support': report[target_name]['support']})
                except Exception as e:
                    summary_file.write(f"=== OneClassSVM Failed (bad fit dimensions?) ===\n\n")
                    skip_roc_plot = True
                    print(e)

                print(f"Node:{node}\t Node global label: {node_global_label}")
                if skip_roc_plot is False:
                    try:
                        roc_auc = auc(fpr, tpr)
                        plot_roc(node, fpr, tpr, roc_auc, output_path, ref_embeds_cfg[0])

                        roc_values.append({'node': node,
                                           'node_global_label': node_global_label,
                                           'classifier': 'OneClassSVM',
                                           'hyperparam1': ref_embeds_cfg[0],
                                           'hyperparam2': -1,
                                           'roc_auc': roc_auc,
                                           })
                    except Exception as e:
                        summary_file.write(f"---> FAILED TO GENERATE ROC CURVE FROM VALUES\n\n")
                        print(e)

        df_detailed_classifier_data = pd.DataFrame(nu_values)
        df_roc_classifier_data = pd.DataFrame(roc_values)

        df_detailed_classifier_data.to_hdf(out_files['df_detailed_classifier_data'].path, key='df', mode='w')
        df_roc_classifier_data.to_hdf(out_files['df_roc_classifier_data'].path, key='df', mode='w')

        summary_file.write('Done')

        return 'succeeded'
