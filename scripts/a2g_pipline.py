from a2g_prediction import a2g_single_TF_prediction
from single_TF_training import single_TF_training
from single_pseudo_train_pred import pseudo_training
from submission import submit
from pseudo_submission import pseudo_submit
import config
from single_TF_prediction import single_TF_prediction


print("---"*100)
print("HTS")
print("---"*100)
target_methods = ["GHTS", "CHS"]
source_method = "HTS"
training_tags = ["first_final"]
TFs = ["CREB3L3", "FIZ1", "GCM1", "MKX", "MSANTD1", "MYPOP", "SP140L",
       "TPRX1", "ZBTB47", "ZFTA", "ZNF286B", "ZNF500", "ZNF780B", "ZNF831", "ZNF721"]

# TFs = ["ZNF721"]

# TFs = ['NFKB1']

# training_tagss = [["cycle1"], ["cycle2"],  ["cycle3"], ["cycle4"]]
# cycle_numss = [["1"], ["2"], ["3"], ["4"]]

# for cn, training_tags in zip(cycle_numss, training_tagss):
       # config.cycle_nums = cn


              # single_TF_training(source_method, TF, augmentations=None,
              #                                                  training_tags=training_tags)
              
       # single_TF_prediction(source_method, TF, training_tags=training_tags)

       # predict_df, predictions_dir, train_tags = submit(source_method, TFs,training_tags=training_tags)
       # predict_df.to_csv(predictions_dir.parent.parent / \
       #                      (source_method + "_" + "_" + \
       #                        "_".join(train_tags) + ".tsv"), 
       #                      sep='\t', index=False)

single_TF_training(source_method, "ZNF721", augmentations=None,
                                          training_tags=training_tags)
for TF in TFs:
    for target_method in target_methods:
        a2g_single_TF_prediction(target_method=target_method, source_method=source_method, target_TF=TF, training_tags=training_tags, 
                                    slice_size=40, step_size=20)


for target_method in target_methods:
    print(target_method, "initial submit")
    predict_df, predictions_dir, training_tags = submit(target_method, TFs, training_tags)
    predict_df.to_csv(predictions_dir.parent.parent / \
                                (target_method + "_" + "_" + \
                                "_".join(training_tags) +"_hts_a2g" + ".tsv"), 
                                sep='\t', index=False)


for target_method in target_methods:
    for TF in TFs:
        pseudo_training(target_method, TF, training_tags)
    

    predict_df, predictions_dir, training_tags = pseudo_submit(target_method, TFs, training_tags)
    pred_dir = predictions_dir.parent.parent / \
                            (target_method + "_" + "_" + \
                              "_".join(training_tags) +"_hts_a2g_pseudo" + ".tsv")
    print(pred_dir)
    predict_df.to_csv(pred_dir, 
                            sep='\t', index=False)
