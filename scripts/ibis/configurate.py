import torch
from pathlib import Path
from ibis.utils.generate_utils.hts import hts_gen_func
from ibis.utils.file_utils.hts import read_hts_train, read_hts_test
from ibis.utils.generate_utils.sms import sms_gen_func
from ibis.utils.file_utils.sms import read_sms_train, read_sms_test
from ibis.utils.generate_utils.pbm import pbm_gen_func
from ibis.utils.file_utils.pbm import read_pbm_train, read_pbm_test
from ibis.utils.generate_utils.ghts import ghts_gen_func
from ibis.utils.file_utils.ghts import read_ghts_test, read_ghts_train

from ibis.utils.file_utils.chs import read_chs_test

from ibis.models.resnet_18a import ResNet1DBinary18
from ibis.models.resnet_18g import resnet18_1d

from ibis.models.resnet_34g import resnet34_1d_binary


def get_data_and_weigths_paths():
    data_path = Path("/home/selivanov/ml_projects/ibis/final_ibis/data/")
    weights_path = Path("/home/selivanov/ml_projects/ibis/final_ibis/weights/")

    return  data_path, weights_path


def get_train_configuration(method_name, target_TF, train_tags):

    data_path, weights_path = get_data_and_weigths_paths()
    train_data_path = data_path / "train"
    
    if method_name == "HTS":
        gen_func = hts_gen_func
        read_func = read_hts_train
        criterion = torch.nn.BCELoss()
        method_path = train_data_path / method_name
        weights_dir_save = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        is_regression = False
        model = ResNet1DBinary18()



    if method_name == "SMS":
        gen_func = sms_gen_func
        read_func = read_sms_train
        criterion = torch.nn.BCELoss()
        method_path = train_data_path / method_name
        weights_dir_save = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        is_regression = False
        model = resnet18_1d()


    if method_name == "PBM":
        quantile = 55
        gen_func = lambda x, y : pbm_gen_func(x, y, quantile=55)
        read_func = lambda x, y : read_pbm_train(x, y, quantile=95)
        criterion = torch.nn.MSELoss()
        method_path = train_data_path / method_name
        weights_dir_save = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        is_regression = True
        model = resnet18_1d()


    if method_name == "GHTS":
        gen_func = ghts_gen_func
        read_func = read_ghts_train
        criterion = torch.nn.BCELoss()
        method_path = train_data_path / method_name
        weights_dir_save = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        is_regression = False
        model = resnet34_1d_binary()




    return  gen_func, read_func, criterion, \
            method_path, weights_dir_save, weights_dir_load, \
            is_regression, model


def get_test_configuration(method_name, target_TF, train_tags):

    
    data_path, weights_path = get_data_and_weigths_paths()
    test_data_path = data_path / "test"
    predictions_path =  data_path / "predictions"

    if method_name == "HTS":

        test_path = test_data_path / f'{method_name}_participants.fasta'
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        read_func = read_hts_test
        predictions_dir = predictions_path / method_name / target_TF / "_".join(train_tags)
        model = ResNet1DBinary18()


    if method_name == "SMS":

        test_path = test_data_path / f'{method_name}_participants.fasta'
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        read_func = read_sms_test
        predictions_dir = predictions_path / method_name / target_TF / "_".join(train_tags) 
        model = resnet18_1d()

        
    if method_name == "PBM":

        test_path = test_data_path / f'{method_name}_participants.fasta'
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        read_func = read_pbm_test
        predictions_dir = predictions_path / method_name / target_TF / "_".join(train_tags) 
        model = resnet18_1d()



    if method_name == "GHTS":

        test_path = test_data_path / f'{method_name}_participants.fasta'
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        read_func = read_ghts_test
        predictions_dir = predictions_path / method_name / target_TF / "_".join(train_tags) 
        model = resnet34_1d_binary()

    
    if method_name == "CHS":

        test_path = test_data_path / f'{method_name}_participants.fasta'
        weights_dir_load = weights_path / method_name / target_TF / "_".join(train_tags) / "best_ResNet18.pth"
        read_func = read_chs_test
        predictions_dir = predictions_path / method_name / target_TF / "_".join(train_tags) 
        model = resnet34_1d_binary()



    return read_func, weights_dir_load, test_path, predictions_dir, model


