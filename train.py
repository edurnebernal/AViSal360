import torch
from dataloader import AV_Dataloader
from audiovisual_sphericalKLD import AV_KLWeightedLoss
import datetime
import os
import time
from torch.utils.data import DataLoader
import model

# Import config file
import config

def read_txt_file(path_to_file):
    """
    The names of the videos to be used for training, they must be in a single line separated
    with ','.
    :param path_to_file: where the file is saved (ex. 'data/file.txt')
    :return: list of strings with the names
    """

    with open(path_to_file) as f:
        for line in f:
            names = line.rsplit('\n')[0].split(',')
    return names

def train(train_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='AViSal360', checkpoint=None):

    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(ckp_path):
        os.mkdir(ckp_path)

    wd = 0.0
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum) 


    # Save in a txt all the configuration parameters in the path folder
    parameters = {'Frames path:': config.frames_dir, 
                  'Audio AEM path:': config.audio_AEM_dir, 
                  'Ground truth path:': config.gt_dir, 
                  'Sequence length:': config.sequence_length, 
                  'Resolution:': config.resolution, 
                  'Batch size:': config.batch_size, 
                  'Epochs:': EPOCHS, 
                  'Learning rate:': lr,
                  'Alpha:': config.alpha,
                  'Optimizer:': optimizer.__class__.__name__,
                  'Momentun:': momentum,
                  'Weight decay:': wd,
                  'Loss:': criterion.__class__.__name__,
                  'model:': model.__class__.__name__}
    
    with open(path + '/parameters.txt', 'w') as f:
        for key, value in parameters.items():
            f.write('%s:%s\n' % (key, value))

    ini_epoch = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ini_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    model.train()

    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []

    # Training loop
    for epoch in range(ini_epoch, EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        counter_train = 0
        print("Epoch: ", epoch)  
        print("Training...") 
        for x, aem, emb, y in train_data:
            
            # Skip if batch size is 1
            if x.shape[0] == 1:
                    continue
            model.zero_grad()
            pred = model(x.to(device), aem.to(device), emb.to(device))

            aem_gt = aem[:, -1, :, :, :]
            # Reshape the ground truth to match the output of the model
            aem_gt = torch.nn.functional.interpolate(aem_gt, size=(config.resolution[0], config.resolution[1]), mode='bicubic', align_corners=False)
            
            loss = criterion(pred[:, 0, :, :], y[:, -1, 0, :, :].to(device), aem_gt[:,0,:,:].to(device))

            loss.sum().backward()
            optimizer.step()

            avg_loss_train += loss.sum().item()

            counter_train += 1

            if counter_train%25==0:
                print("Step {} , ------------------ Loss: {}".format(counter_train, avg_loss_train / counter_train))

        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))
        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
            
        if epoch % 50 == 0:
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model_' + str(epoch) + '.pt')
    
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model_final_' + str(epoch) + '.pt')

    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Print the name of the device
    print(torch.cuda.get_device_name(device))

    k_folds = os.listdir(config.k_folds_dir)
    
    for fold in k_folds:
        # Create the model
        avisal360 = model.AVSal_IB4LconAEM_2SP(input_dim=3, hidden_dim=config.hidden_dim, output_dim=1)

        videos_train_file = os.path.join(config.k_folds_dir, fold)

        loss = AV_KLWeightedLoss(alpha=config.alpha, AUC=False)

        video_names_train = read_txt_file(videos_train_file)
        train_video360_dataset = AV_Dataloader(config.frames_dir, config.gt_dir,
                                            path_to_embeddings= config.embeddings_dir, 
                                            path_to_AEM=config.audio_AEM_dir, 
                                            video_names=video_names_train, 
                                            frames_per_data=config.sequence_length, 
                                            split ='all', 
                                            HxW_frames = config.resolution,
                                            HxW_AEM =(120, 160),
                                            inter_seq=True,
                                            window= 18,
                                            threading=True)

        train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=20, shuffle=True)

        print(avisal360)
        model = train(train_data, avisal360, device, loss, lr=config.lr, EPOCHS=config.epochs,model_name='AVIONES_sinOF_a075_e120' + fold, checkpoint=None)
        print("Training finished")