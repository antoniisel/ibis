from single_TF_prediction import single_TF_prediction
from single_TF_training import single_TF_training
from submission import submit
from ibis.utils.augmentations import insert_N_w_size, insert_N,  mask_last_with_N, \
                                          insert_base, mask_first_and_last_with_N, \
                                          mask_N_with_position,\
                                              get_complement, get_reverse_compliment,\
                                                flip
                                                        
import config

print(config.cycle_nums)

# method_name = "GHTS"
# TFs = ["GABPA", "PRDM5", "ZNF362", "ZNF407"]
# training_tags=[""]


# for TF in TFs:
#     single_TF_training(method_name, TF, augmentations=[lambda x:  get_complement(x, p=0.3),
                                                       
#                                                        ],
#                                                            training_tags=training_tags)
#     single_TF_prediction(method_name, TF, training_tags=training_tags)


# TFs = ["GABPA", "PRDM5", "ZNF362", "ZNF407"]
# predict_df, predictions_dir, train_tags = submit(method_name, TFs,training_tags=training_tags)
# predict_df.to_csv(predictions_dir.parent.parent / \
#                             (method_name + "_" + "_" + \
#                               "_".join(train_tags) + ".tsv"), 
#                             sep='\t', index=False)