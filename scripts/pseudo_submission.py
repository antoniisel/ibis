import pandas as pd
import numpy as np
import config
from ibis.configurate import get_test_configuration

def pseudo_submit(method_name, TFs, training_tags=["default"]):

    predict_tfs = []
    for target_TF in TFs:
        print(target_TF)
        read_func, weights_dir_load, test_path, predictions_dir, model = get_test_configuration(method_name, target_TF, training_tags)
        preds = np.load(predictions_dir / "predictions_pseudo.npy").flatten()
        tags = np.load(predictions_dir / "tags.npy")
        predict_tfs.append(preds)
    
    predict_tfs = np.array(predict_tfs)
    round_pred = np.vectorize(lambda x: round(x, 5))
    predict_tfs = round_pred(predict_tfs)
    d1 = {tf:predict_tf for tf, predict_tf in zip(TFs, predict_tfs)}
    d2 = {"tags":tags}
    z = dict(list(d2.items()) + list(d1.items()))
    predict_df = pd.DataFrame(z)

    return predict_df, predictions_dir, training_tags


if __name__ == "__main__":
    method_name = "GHTS"
    # TFs = ["LEF1", "NACC2", "RORB", "TIGD3"]
    # TFs = ["GABPA", "PRDM5", "ZNF362", "ZNF407"]
    TFs = ["NFKB1"]
    training_tags=["first_final"]

    predict_df, predictions_dir, training_tags = pseudo_submit(method_name, TFs, training_tags)

    pred_dir = predictions_dir.parent.parent / \
                            (method_name + "_" + "_" + \
                              "_".join(training_tags) +"_hts_a2g_pseudo" + ".tsv")
    print(pred_dir)
    predict_df.to_csv(pred_dir, 
                            sep='\t', index=False)
    


    
