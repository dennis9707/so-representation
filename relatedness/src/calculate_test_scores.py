import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve, matthews_corrcoef
def calculate_score(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    predtions = [l.strip().split('\t')[1] for l in data[1:]]

    test_df = pd.read_csv('../../../data/relate/LinkPrediction_Dataset/processed_test.csv')
    gold_test = test_df['label']

    label2id = { '"duplicate"':0, '"direct"':1, '"indirect"':2, '"isolated"':3 }
    #label2id = { 'duplicate':0, 'non-duplicate':1}
    gold_label_id = [label2id[g] for g in list(gold_test)]
    #print(gold_label_id)
    pred_label_id = [label2id[p] for p in predtions]
    #print(pred_label_id)

    f1 = f1_score(y_true=gold_label_id, y_pred=pred_label_id, average='micro')
    rec = recall_score(y_true=gold_label_id, y_pred=pred_label_id, average='micro')
    prec = precision_score(y_true=gold_label_id, y_pred=pred_label_id, average='micro')
    acc = accuracy_score(y_true=gold_label_id, y_pred=pred_label_id)

    print('Acc:', acc)
    print(' F1:{}  Recall:{} Precision:{} '.format(f1, rec, prec))

    ## Dup, direct, indirect, Ioslated
    gold_label_id_dup = [ 1 if l==0 else 0  for l in gold_label_id]
    pred_label_id_dup = [ 1 if l==0 else 0  for l in pred_label_id] 
    gold_label_id_direct = [ 1 if l==1 else 0  for l in gold_label_id]
    pred_label_id_direct = [ 1 if l==1 else 0  for l in pred_label_id]
    gold_label_id_indirect = [ 1 if l==2 else 0  for l in gold_label_id]
    pred_label_id_indirect = [ 1 if l==2 else 0  for l in pred_label_id]
    gold_label_id_isolated = [ 1 if l==3 else 0  for l in gold_label_id]
    pred_label_id_isolated = [ 1 if l==3 else 0  for l in pred_label_id]
    #print(pred_label_id_dup)
    f1_dup = f1_score(y_true=gold_label_id_dup, y_pred=pred_label_id_dup)
    f1_direct = f1_score(y_true=gold_label_id_direct, y_pred=pred_label_id_direct)
    f1_indirect = f1_score(y_true=gold_label_id_indirect, y_pred=pred_label_id_indirect)
    f1_isolated = f1_score(y_true=gold_label_id_isolated, y_pred=pred_label_id_isolated)
    print(' Duplicated:{}  Direct:{} Indirect:{} Isolated:{}'.format(f1_dup, f1_direct, f1_indirect, f1_isolated))

def main():
    calculate_score("../result/April-21-longformer-code-cl/predict_results.txt")
if __name__ == "__main__":
    main()