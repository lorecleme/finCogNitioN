# remember to import the CNN architecture and instantiate the model

loss = nn.CrossEntropyLoss()

criterion = optim.AdamW(model.parameters(), lr=0.00001)


def train(model, loss_f, optim, dataloader):
    
    model.train()
    train_losses = list()
    
    train_loss = 0
    n_batches = len(dataloader)
    
    for i, batch in enumerate(tqdm(dataloader)):
        
        
        imgs, labels = batch[0], batch[1]
        
        labels  = labels.type(torch.LongTensor)
        
        imgs, labels = imgs.cuda(), labels.cuda()
        
        
        ## -- forward pass -- ##
        
        preds = model(imgs)
        
        
        ## -- computing the loss wrt G.T. -- ##
        loss = loss_f(preds, labels)
        
        
        ## -- Backpropagation -- ##
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        
        train_losses.append(loss.item())
        train_loss += loss.item()
        
    print(f"Average TRAIN Loss :: {train_loss/n_batches} ")
    
    return train_loss/n_batches


def test(model, iteratore, loss_f):
    num_batches = len(iteratore)
    test_loss = 0
    accuracy = 0
    
    test_losses = list()
    
    model.eval()
    
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iteratore)):

            img, labels = batch[0], batch[1]
            labels  = labels.type(torch.LongTensor)
            img, labels = img.cuda(), labels.cuda()

            pred = model(img)
            loss = loss_f(pred, labels)
            test_loss += loss.item()
            
            
            pred_indices = torch.argmax(pred, dim=1)
            label_test = labels.cpu().float()
            pred_indices = pred_indices.cpu().float()
            distances = torch.where(pred_indices == label_test, torch.tensor(1), torch.tensor(0))
            distances = distances.numpy()
            accuracy += sum(distances)/len(label_test)
            
            
            test_losses.append(loss.item())
            
    test_loss = test_loss / num_batches
    accuracy_tot = accuracy / num_batches
    
    print(f"Average TEST loss: {test_loss}, TEST accuracy: {accuracy_tot*100}%")
    return test_loss



#min_loss = 100

#train_losses = list()
#test_losses = list()

for epoch in range(5):
    print(f"----- epoch {epoch} -----")
    train_losses.append(train(model, loss, criterion, train_loader))
    t_loss = test(model, val_loader, loss)
    
    test_losses.append(t_loss)
    
    if t_loss < min_loss:
        min_loss = t_loss
        torch.save(model, 'model_checkpoint.pth') # actually this could be saved using the timestamp
    print("\n")
