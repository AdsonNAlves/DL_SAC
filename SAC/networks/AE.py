import torch
from torch import nn, optim
import numpy as np
import os
import csv

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        #Encoder
        self.encoder_hidden_1 = nn.Linear(64*64, 32*32)
        self.encoder_hidden_2 = nn.Linear(32*32, 16*16)
        self.encoder_hidden_3 = nn.Linear(16*16, 8*8)
        self.encoder_hidden_4 = nn.Linear(8*8, 4*4)
        self.encoder_hidden_5 = nn.Linear(4*4, 2*2)
        #self.encoder_hidden_6 = nn.Linear(2*2, 1*1)

        #Decoder
        #self.decoder_hidden_1 = nn.Linear(1*1, 2*2)
        self.decoder_hidden_1 = nn.Linear(2*2, 4*4)
        self.decoder_hidden_2 = nn.Linear(4*4, 8*8)
        self.decoder_hidden_3 = nn.Linear(8*8, 16*16)
        self.decoder_hidden_4 = nn.Linear(16*16, 32*32)
        self.decoder_hidden_5 = nn.Linear(32*32, 64*64)


    def forward(self, features):

        #code
        activation = torch.relu(self.encoder_hidden_1(features))
        activation = torch.relu(self.encoder_hidden_2(activation))
        activation = torch.relu(self.encoder_hidden_3(activation))
        activation = torch.relu(self.encoder_hidden_4(activation))
        code = torch.relu(self.encoder_hidden_5(activation))
        #code = torch.relu(self.encoder_hidden_6(activation))

        #decodea
        activation = torch.relu(self.decoder_hidden_1(code))
        activation = torch.relu(self.decoder_hidden_2(activation))
        activation = torch.relu(self.decoder_hidden_3(activation))
        activation = torch.relu(self.decoder_hidden_4(activation))
        reconstructed = torch.relu(self.decoder_hidden_5(activation))
        #reconstructed = torch.relu(self.decoder_hidden_6(activation))

        return code, reconstructed

    def execute_ae(self):

        step= int(len(self.origin_image)/(self.image_size*self.image_size))
        ac=[]

        for i in range(step):
            ac.append(self.origin_image[
                      (self.image_size * self.image_size) * i:(self.image_size * self.image_size) * (i + 1)].reshape(
                self.image_size, self.image_size))

        #ac = torch.from_numpy(np.array(ac).astype('float32'))
        train = torch.utils.data.DataLoader(ac, batch_size=4, shuffle=True)
        self.loss_ae = 1
        self.epoch = 0
        while self.loss_ae > 0.004 and self.epoch <30:
            self.loss_ae = 0
            for batch_images in train:
                self.autoencoder_optimizer.zero_grad()
                batch_images = batch_images.view(-1,self.image_size*self.image_size).to(self.device)
                self.code, decoder = self.autoencoder(batch_images)
                train_loss_AE = self.autoencoder_criterion(decoder, batch_images)
                train_loss_AE.backward()
                self.autoencoder_optimizer.step()
                self.loss_ae += train_loss_AE.item()

            self.loss_ae = self.loss_ae / len(train)
            self.epoch +=1

        # print('Epoch AE {} | Precision AE {:.6f}'.format(self.epoch, 1- self.loss_ae))
        self.code = self.code.detach().numpy()


    def start_ae(self,first_obs,origin_image,device,image_size,autoencoder_lr,restore_path):

        self.origin_image = origin_image

        if first_obs:
            self.device = device
            self.image_size = image_size
            self.autoencoder_lr = autoencoder_lr
            self.restore_path = restore_path

            self.autoencoder = AE().to(self.device)
            self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(),self.autoencoder_lr)
            self.autoencoder_criterion = nn.MSELoss()

            self.save_ae =0
            try:
                checkpoint = torch.load(self.restore_path + 'autoencoder.pt')
                self.autoencoder.load_state_dict(checkpoint['autoencoder'])
                self.autoencoder_optimizer.load_state_dict(checkpoint['autoencoder_optimizer'])
                self.epoch = checkpoint['epoch']
                self.loss_ae = checkpoint['loss_ae']
                self.autoencoder.eval()
            except:
                None
                # print('Não foi possível carregar um modelo AE pré-existente')

            # Preparing log csv to AE
            if not os.path.isfile(os.path.join(self.restore_path, 'progress_ae.csv')):
                print('There is no csv to AE there')
                with open(os.path.join(self.restore_path, 'progress_ae.csv'), 'w') as aecsv:
                    writer = csv.writer(aecsv, delimiter=';',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["Episode", "Loss_ae"])

        self.execute_ae()
        self.save_ae += 1
        if self.save_ae % 51 == 0:
            # save model AE
            torch.save({'epoch': self.epoch,
                        'loss_ae': self.loss_ae,
                        'autoencoder': self.autoencoder.state_dict(),
                        'autoencoder_optimizer': self.autoencoder_optimizer.state_dict()
                        },self.restore_path+'autoencoder.pt')
            # print('saving AE model at with loss {:.6f}'.format(self.loss_ae))
            #save data AE

            with open(os.path.join(self.restore_path, 'progress_ae.csv'), 'a') as csvfile:
                rew_writer = csv.writer(csvfile, delimiter=';',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

                rew_writer.writerow([self.epoch,
                                     self.loss_ae])
            self.save_ae=0

env= AE()
def set_image(**kwargs):

    try:
        first_obs = kwargs["first"]
        origin_image = kwargs["origin_image"]
        device = kwargs["device"]
        image_size = kwargs["image_size"]
        autoencoder_lr = kwargs["lr_rate"]
        restore_path = kwargs["restore_path"]
        env.start_ae(first_obs,origin_image,device,image_size,autoencoder_lr,restore_path)
    except:
        print("Ainda sem imagem")
        env.code = [np.zeros(4).astype("float32"),np.zeros(4).astype("float32"),
                    np.zeros(4).astype("float32"),np.zeros(4).astype("float32")]
        env.loss_ae = 1
    return env.code, env.loss_ae