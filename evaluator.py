import argparse
import numpy as np
from sklearn.metrics import classification_report

def eval1(pred_path,true_path):
    with open(pred_path,'r') as f:
        pred=[line.split('\t') for line in f]
        pred= {item[0]:int(item[1][0]) for item in pred}
    with open(true_path,'r') as f:
        true=[line.split('\t') for line in f]
        true= {item[0]:int(item[1][0]) for item in true}

    y_pred,y_true=[],[]
    for k,v in pred.items():
        y_pred.append(v)
        try:
            y_true.append(true[k])
        except KeyError:
            print(f"key not the same,{k}in pred file but not true file" )
    print(classification_report(y_true, y_pred))

    return



def eval2(pred_path,true_path):
    with open(pred_path,'r') as f:
        pred=[line.split('\t') for line in f]
        pred= {item[0]:float(item[1][:-1]) for item in pred}
    with open(true_path,'r') as f:
        true=[line.split('\t') for line in f]
        true= {item[0]:float(item[1][:-1]) for item in true}

    y_pred,y_true=[],[]
    for k,v in pred.items():
        y_pred.append(v)
        try:
            y_true.append(true[k])
        except KeyError:
            print(f"key not the same,{k}in pred file but not true file" )
    print(np.corrcoef(y_pred,y_true)[0][1])
    return np.corrcoef(y_pred,y_true)[0][1]




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset & model paths
    parser.add_argument('--pred_file', type=str, required=True,
                        help='path to the predicted file')
    parser.add_argument('--true_file', type=str, required=True,
                        help='Path to the true file.')
    parser.add_argument('--task', type=str, required=True,
                        help='task type', choices=['task1','task2'])

    args = vars(parser.parse_args())
    print(args)
    
    if args['task']=='task1':
        eval1(args['pred_file'],args['true_file'])

    elif args['task']=='task2':
        eval2(args['pred_file'],args['true_file'])

    else:
        print("Please provide a valid task name!")
