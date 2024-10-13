import numpy as np
import cv2 as cv
import os
import copy
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from maxMin import maxAndMin
from conv2Net import ConvNet
from torch.optim.lr_scheduler import CyclicLR
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

torch.manual_seed(42)

def dataLoad(path, want=0):
    nameList = os.listdir(path)

    try:
        nameList.remove(".DS_Store")
    except:
        pass

    totalHolder = []
    dims = [1600, 900]

    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append((torch.tensor([[im]]).to(dtype=torch.float, device=device) / top,
                             torch.tensor([[int((name.split("."))[want]) / dims[want]]]).to(dtype=torch.float, device=device)))

    return totalHolder

def create_dataloader(data, batch_size, shuffle=True, seed=2):
    if shuffle and seed is not None:
        np.random.seed(seed)  

    indices = np.arange(len(data))  
    if shuffle:
        np.random.shuffle(indices)  

    shuffled_data = [data[i] for i in indices]
    
    images, labels = zip(*shuffled_data)
    images = torch.cat(images)  # Concatenar las imágenes
    labels = torch.cat(labels)  # Concatenar las etiquetas

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluateModel(model, dataloader_derecho, dataloader_izquierdo, sidelen=1600):
    model.eval()
    err = 0
    total_samples = 0  # Contador para el número total de muestras

    
    for (im1, label1), (im2, label2) in zip(dataloader_derecho, dataloader_izquierdo):
            im1 = im1.to(device)
            im2 = im2.to(device)
            label1 = label1.to(device)  # Asegúrate de que las etiquetas también están en el dispositivo

            output = model(im1, im2)  # Usando ambas imágenes como entradas

            # Calcular el error absoluto promedio para el minibatch
            batch_err = torch.abs(output - label1).sum().item()  # Sumar el error absoluto
            err += batch_err
            total_samples += label1.size(0)  # Sumar el número de muestras en el minibatch

    model.train()
    return ((err / total_samples) * sidelen)  # Promedio del error

# Cargar datos
trainingSet_Derecho = dataLoad("ojoderecho")
trainingSet_Izquierdo = dataLoad("ojoizquierdo")
test_Izquierdo = dataLoad("testeyesizquierdo")
test_Derecho = dataLoad("testeyesderecho")

# Crear DataLoaders
batch_size = 32  # Ajusta el tamaño del minibatch aquí
train_loader_derecho = create_dataloader(trainingSet_Derecho, batch_size)
train_loader_izquierdo = create_dataloader(trainingSet_Izquierdo, batch_size)
test_loader_derecho = create_dataloader(test_Derecho, batch_size)
test_loader_izquierdo = create_dataloader(test_Izquierdo, batch_size)

num_epochs = 30
early_stopping_patience = 5  # Número de épocas para esperar antes de aplicar "early stopping"

def trainModel():
    early_stopping_counter2 = 0
    model = ConvNet().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []
    trainlossscore = []
    testlossscore = []

    model.train()

    for epoch in range(num_epochs):
        if early_stopping_counter2 == early_stopping_patience:
             break
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        for (im1, label1), (im2, label2) in zip(train_loader_derecho, train_loader_izquierdo):
            optimizer.zero_grad()
            im1 = im1.to(device)
            im2 = im2.to(device)
            output = model(im1, im2)
            #print("Este es el label1" + str(label1))
            #print("Este es el label1" + str(label2))
            #if 1==1:
                #exit()
            loss = criterion(output, label1)
            loss.backward()
            optimizer.step()

            
            
        if epoch +1 %  5==0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            #print(f'Current Learning Rate: {current_lr:.6f}')
            #print(f'Learning Rate: {scheduler.get_last_lr()}')

    
        testSc = evaluateModel(model, test_loader_derecho, test_loader_izquierdo)  # Pasar ambos loaders
        trainSc = evaluateModel(model, train_loader_derecho, train_loader_izquierdo)

        if testSc < bestScore:
                bestModel = copy.deepcopy(model)
                bestScore = testSc
                early_stopping_counter2 = 0  # Reiniciar el contador si hay mejora
        else:
                early_stopping_counter2 += 1

        testscores.append(testSc)
        trainscores.append(trainSc)
        trainlossscore.append(loss.item())

        print(f'Train Accuracy: {trainSc:.4f}')
        print(f'Test Accuracy: {testSc:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}  ')
    
    # Guardar resultados
    df = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Train Accuracy': trainscores,
        'Test Accuracy': testscores,
        'Train loss': trainlossscore,
    })
    df.to_excel('results.xlsx', index=False)

    finalScore = evaluateModel(bestModel, test_loader_derecho,test_loader_izquierdo)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore)) + ".plt")


trainModel()
