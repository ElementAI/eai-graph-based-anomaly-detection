import os.path as osp
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
from sklearn.metrics import auc, confusion_matrix, classification_report
from eai_graph_tools.graph_tools.exporters.pandas.export_h5 import export_dataframe_to_h5
from eai_graph_tools.airflow_tasks.model_tasks import InferenceOutput
from eai_graph_tools.datasets.intervals import load_dataset_vendor


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


def get_labeled_embeddings(metrics_df, node, start, end):

    df_embeddings_subset = metrics_df.loc[start:end]
    df_embeddings_subset = df_embeddings_subset.loc[df_embeddings_subset['node'] == node]

    labeled_data = {
        "embeddings": [],
        "class_label": [],
        "class_label_int": [],
    }

    for index, row in df_embeddings_subset.iterrows():
        labeled_data["embeddings"].append(
            np.fromstring(str(row['embeddings']).replace('[', '').replace(']', ''),
                          dtype=float,
                          sep=','))
        labeled_data["class_label"].append(row['class_label'])
        labeled_data["class_label_int"].append(1 if row['class_label'] == 'BENIGN' else -1)

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

    for k, interval in tqdm(enumerate(inference_dataset)):
        interval_start = inference_dataset.start + k * inference_dataset.interval_width
        interval_end = inference_dataset.start + (k + 1) * inference_dataset.interval_width
        interval_info = inference_dataset.get_interval(k)

        for noi in nodes_of_interest:
            training_intervals_limit = int(training_intervals_count) - 1

            if noi in interval_info \
                    and not np.isnan(baselines[noi]).any() \
                    and not np.isnan(node_embeddings[noi][k]).any()\
                    and len(node_embeddings[noi][:training_intervals_limit]) != 0:
                if generate_cosine_only:
                    row_values = {
                        'node': noi,
                        'baseline_embeddings': np.array2string(baselines[noi], separator=', '),
                        'embeddings': np.array2string(node_embeddings[noi][k], separator=', '),
                        'class_label': interval_info[noi],
                        'timestamp_original': interval_info['timestamp_original'],
                        'timestamp_adjusted': interval_info['timestamp_adjusted'],
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
                        'class_label': interval_info[noi],
                        'timestamp_original': interval_info['timestamp_original'],
                        'timestamp_adjusted': interval_info['timestamp_adjusted'],
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

                df_embeddings = df_embeddings.append(row_values, ignore_index=True)
    return df_embeddings


@attr.s(auto_attribs=True)
class CreatePostProcessingDFOutput():
    df1_detailed_classifier_data: pd.DataFrame = attr.ib(default=None)
    df2_roc_classifier_data: pd.DataFrame = attr.ib(default=None)
    metrics_df: pd.DataFrame = attr.ib(default=None)
    output_path: os.PathLike = attr.ib(default=".")
    experiment_name: str = attr.ib(default="0")

    def save(self):
        output_file = os.path.join(self.output_path, f'analysis_{self.experiment_name}.h5')
        self.df1_detailed_classifier_data.to_hdf(output_file, key='df1_detailed_classifier_data', mode='w')
        self.df2_roc_classifier_data.to_hdf(output_file, key='df2_roc_classifier_data', mode='w')
        self.metrics_df.to_hdf(output_file, key='metrics_df', mode='w')

        # Saving df standalone as well to make it easier to visualize with dfgui
        output_file = os.path.join(self.output_path, f'df1_detailed_classifier_data.h5')
        export_dataframe_to_h5(self.df1_detailed_classifier_data, output_file)
        output_file = os.path.join(self.output_path, f'df2_roc_classifier_data.h5')
        export_dataframe_to_h5(self.df2_roc_classifier_data, output_file)


def post_analysis(tp, in_files, out_files, *op_args, **op_kwargs):

    # todo: add as config params...
    svm_kernel = 'rbf'
    svm_gamma = 'scale'

    start = tp['start']
    end = tp['end']

    with open(out_files['metrics_summary_file'].path, "w") as summary_file:

        summary_file.write(f"\nMetrics data {tp['experiment_name']}\n\n")

        output_path = osp.join(Variable.get(tp['airflow_vars']['out_dir']), Variable.get(tp['airflow_vars']['hash']))

        inf_output = InferenceOutput(filename=in_files['node_embeddings'].path)
        inf_output.restore()
        inference_dataset = load_dataset_vendor(in_files,
                                                start,
                                                tp['interval_width'])

        training_intervals_count = Variable.get(tp['airflow_vars']['training_intervals_count'], default_var=0)

        # Generating metrics for selected nodes
        metrics_df = create_metrics_df(inf_output.node_embeddings,
                                       inference_dataset,
                                       tp['nodes_of_interest'],
                                       training_intervals_count,
                                       generate_cosine_only=True)

        df = metrics_df
        df['index'] = df['timestamp_adjusted']
        df.set_index('index', inplace=True)

        # Preparing reference_embeddings
        embeds_192_168_10_50, _ = get_labeled_embeddings(df, '192.168.10.50', start, end)
        file_path_ref_embeds_192_168_10_50 = os.path.join(output_path, "ref_embeds_192_168_10_50.p")
        os.makedirs(output_path, exist_ok=True)
        pickle.dump(embeds_192_168_10_50['embeddings'], open(file_path_ref_embeds_192_168_10_50, "wb"))

        inferred_reference_embeddings = []
        for node in tp['reference_nodes']:
            labeled_data, _ = get_labeled_embeddings(df, node, start, end)
            inferred_reference_embeddings.extend(labeled_data['embeddings'])
        file_path_ref_nodes_embeds = os.path.join(output_path, "ref_nodes_embeds.p")
        pickle.dump(inferred_reference_embeddings, open(file_path_ref_nodes_embeds, "wb"))

        target_names = ['class MALICIOUS', 'class BENIGN']  # Confirm class ordering
        target_names_report = ['class MALICIOUS', 'class BENIGN', 'weighted avg']  # Confirm class ordering

        nu_values = []
        roc_values = []
        for node in tp['nodes_of_interest']:

            # Get test set's embeddings
            x_testing_embeddings, _ = get_labeled_embeddings(df, node, start, end)
            x_test = x_testing_embeddings["embeddings"]
            y_true_test = x_testing_embeddings["class_label_int"]

            # Determine the node label for the entire interval
            node_global_label = 'malicious' if len(set(y_true_test)) > 1 else 'benign'
            # len > 1 means the set contains more than 1 unique value, thus bening and malicious exist.
            summary_file.write(f"NODE: {node}\t Global Label: {node_global_label}\n")

            # Generate a few reference embeddings
            reference_embeddings_configs = []
            # 1- Embeddings from the node itself only (works well on victim nodes)
            ref_embeds_1 = x_test
            reference_embeddings_configs.append(('ref1', ref_embeds_1))     # itself only

            # 2- Embeddings from the node itself + a victim node
            ref_embeds_2 = pickle.load(open(file_path_ref_embeds_192_168_10_50, "rb"))
            ref_embeds_2.extend(x_test)
            reference_embeddings_configs.append(('ref2', ref_embeds_2))     # _embeds_2_itself_plus_1victim

            # 3- Embeddings from the node + all_victims + a few non_victims
            ref_embeds_3 = pickle.load(open(file_path_ref_nodes_embeds, "rb"))
            reference_embeddings_configs.append(('ref3', ref_embeds_3))     # _embeds_3_ref_nodes_list

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

        metrics_df.to_hdf(out_files['df_metrics'].path, key='metrics_df', mode='w')
        df_detailed_classifier_data.to_hdf(out_files['df_detailed_classifier_data'].path, key='df', mode='w')
        df_roc_classifier_data.to_hdf(out_files['df_roc_classifier_data'].path, key='df', mode='w')

        summary_file.write('Done')

        return 'succeeded'
