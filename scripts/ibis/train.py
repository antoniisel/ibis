import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization
from ibis.utils.preprocessing import preprocess_pbm
from ibis.utils.general import add_val_metrics_to_writer, save_weights



def train(model, 
          optimizer,
          criterion,
          num_epochs,
          train_data,
          test_data,
          device,
          batch_size,
          is_regression=False,
          weights_dir_save=None):
    

    print("Start Training")
    writer = SummaryWriter()

    best_score = float('-inf')
    num_iterations = 0
    test_data.clean_data()
    train_data.clean_data()

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) 
    for epoch in range(num_epochs):
        train_data.generate_data()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        # Validation       
        epoch_loss, accuracy, precision, recall, f1, roc_auc = validate(model, test_loader, criterion, epoch, num_epochs, device)        
        # Write metrics to TensorBoard
        add_val_metrics_to_writer(writer, epoch, epoch_loss, accuracy, precision, recall, f1, roc_auc)

        if is_regression:
            target_metrics = -epoch_loss
        else:
            target_metrics = roc_auc

        if target_metrics > best_score:
            best_score = target_metrics
            print("Saving weights with best score =", best_score)
            save_weights(model, weights_dir_save)


        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"epoch № {epoch + 1}/{num_epochs}, Train"):
            num_iterations += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # Compute average training loss
        epoch_loss = running_loss / len(train_loader.dataset)

        # Write loss to TensorBoard
        print(f'Train Loss: {epoch_loss:.4f}')
        writer.add_scalar('Loss/Train', epoch_loss, epoch)


    ## "Last Validation!"
    epoch_loss, accuracy, precision, recall, f1, roc_auc = validate(model, test_loader, criterion, epoch+1, num_epochs, device)
    # Write metrics to TensorBoard
    add_val_metrics_to_writer(writer, epoch, epoch_loss, accuracy, precision, recall, f1, roc_auc)


    if is_regression:
        target_metrics = -epoch_loss
    else:
        target_metrics = roc_auc
    if target_metrics > best_score:
        best_score = target_metrics
        print("Saving weights with best score =", best_score)
        save_weights(model, weights_dir_save)

    print("Training process finished!")
    # Close TensorBoard SummaryWriter
    writer.close()






def validate(model, test_loader, criterion, epoch, num_epochs, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        all_preds, all_labels = [], []
        for inputs, labels in tqdm(test_loader, desc=f"epoch № {epoch}/{num_epochs}, Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            running_loss += loss.item() * inputs.size(0)

            preds = torch.round(outputs).cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(torch.round(labels).cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(test_loader.dataset)
    try :
        roc_auc = roc_auc_score(all_labels, all_preds) 
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f'Epoch [{epoch}/{num_epochs}], Val Loss {epoch_loss:.4f}, '
        f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
        f'Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}')
        
    except ValueError:
        print(f'Epoch [{epoch}/{num_epochs}], Val Loss {epoch_loss:.4f}, ')
        accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0

    return epoch_loss, accuracy, precision, recall, f1, roc_auc, 

