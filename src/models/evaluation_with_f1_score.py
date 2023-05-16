# This file should be used so that users can give just a list of images as input and the function gives
## as output the predicted categories followed by the plots of images/predictions



from sklearn.metrics import f1_score
def test_with_f1(model, iteratore, loss_f):
    LABELS = list()
    PREDICTIONS = list()
    
    
    num_batches = len(iteratore)
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for img, labels in tqdm(iteratore):
            labels  = labels.type(torch.LongTensor)
            img, labels = img.cuda(), labels.cuda()

            pred = model(img)
            test_loss += loss_f(pred, labels).item()
            
            
            pred_indices = torch.argmax(pred, dim=1)
            label_test = labels.cpu().float()
            pred_indices = pred_indices.cpu().float()
            distances = torch.where(pred_indices == label_test, torch.tensor(1), torch.tensor(0))
            distances = distances.numpy()
            accuracy += sum(distances)/len(label_test)
            
            PREDICTIONS += pred_indices.tolist()
            LABELS += labels.tolist()
            
    print(f"F1 Score, None = {f1_score(LABELS, PREDICTIONS, average=None)}")
    print(f"F1 Score, micro = {f1_score(LABELS, PREDICTIONS, average='micro')}")
    print(f"F1 Score, macro = {f1_score(LABELS, PREDICTIONS, average='macro')}")
    print(f"F1 Score, avg = {f1_score(LABELS, PREDICTIONS, average='weighted')}")

    test_loss = test_loss / num_batches
    accuracy_tot = accuracy / num_batches
    print(f"Average TEST loss: {test_loss}, TEST accuracy: {accuracy_tot*100}%")
    return test_loss

test_with_f1(model, val_loader, loss)