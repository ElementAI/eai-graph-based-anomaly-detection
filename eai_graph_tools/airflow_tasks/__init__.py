TP = 0
TN = 1
FP = 2
FN = 3
NODE_ABSENT = 4
P = 5  # positive - no ground truth
N = 6  # negative - no ground truth

BENIGN = 1
MALICIOUS = -1
UNLABELLED = 0


def get_all_nodes_from_dataset(inference_dataset):
    print("Node list not specified: running prediction on entire node list (might be slow!!)")
    all_nodes = []
    for data in inference_dataset:
        interval_node_list = list(data.node_indexes_in_tensors.keys())
        for node in interval_node_list:
            if node not in all_nodes:
                all_nodes.append(node)
    return all_nodes


def get_result_from_prediction_and_ground_truth(pred_label, truth_label):
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

    return result
