import os
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torchvision import transforms
from torch.utils.data import DataLoader
import dataloader as dl
from networks import *
from util import *
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment()
PATH = 'sacred/bin20/'
ex.observers.append(FileStorageObserver.create(PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ex.config
def config():
    tr_conf = {
        'num_classes':42,
        'border_pixels':20,
        'bin_size':1,
        'n_epoch': 400,
        'b_s': 4,
        'n_workers': 4,
        'optimizer': 'Adam',
        'reduction': 'Mean',
        'lr': 1e-5,
        'starting_epoch':0,
        'meta_train_path': '/ds2/YoutubeVOS2018/train/meta.json',
        'im_train_path': '/ds2/YoutubeVOS2018/train/JPEGImages/',
        'ann_train_path': '/ds2/YoutubeVOS2018/train/Annotations/',
        'affine_info': {
            'angle': range(-20, 20),
            'translation': range(-10, 10),
            'scale': range(75, 125),
            'shear': range(-10, 10)},
        'hflip': True,
        'lambda1': 0.6,
        'lambda2': 0.2,
        'lambda3': 0.2
    }


@ex.capture()
def train(initializer,
          encoder,
          decoder,
          convlstm_cell,
          convlstm_middle,
          convlstm_3,
          optimizer,
          dataloader,
          epoch,
          tr_conf,
          train_all=False,
          loss_avg=True):

    initializer.train()
    encoder.train()
    decoder.train()
    convlstm_cell.train()
    convlstm_middle.train()
    convlstm_3.train()

    loss_meter = AverageMeter()
    loss_fn, distance_loss = JaccardIndexLoss(), nn.CrossEntropyLoss()
    loss_list = []

    with tqdm(total=len(dataloader.dataset)) as progress_bar:
        for sequence in dl.pooled_batches(dataloader):
            input, gt, distanc_class = sequence['image'], sequence['gt'], sequence['dists']
            init_frame = torch.cat([input[0], gt[0]], dim=1).to(device)

            if len(input) == 1: continue
            states = initializer(init_frame)
            h, c = states[0]
            h_middle, c_middle = states[1]
            h_3, c_3 = states[2]

            for ii in range(1, len(input)):
                enc, mead_feats = encoder(input[ii].to(device))

                h, c = convlstm_cell(enc, (h, c))
                h_middle, c_middle = convlstm_middle(mead_feats[3], (h_middle, c_middle))
                h_3, c_3 = convlstm_3(mead_feats[2], (h_3, c_3)) 

                decoded_imgs, distance_scores = decoder(h, mead_feats, h_middle, h_3)
                loss = tr_conf['lambda3'] * loss_fn(decoded_imgs, gt[ii].to(device)) + \
                       tr_conf['lambda2'] * distance_loss(distance_scores, distanc_class[ii].squeeze(1).long().to(device)) + \
                       tr_conf['lambda1'] * class_balanced_cross_entropy_loss(decoded_imgs, gt[ii].to(device)) 

                loss_list.append(loss)


            loss_total = sum(loss_list) / len(loss_list) if loss_avg else sum(loss_list)
            del loss_list[:]
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(loss_total.item(), tr_conf['b_s'])
            progress_bar.set_postfix(loss_avg=loss_meter.avg, lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(tr_conf['b_s'])

    if os.access(PATH, os.W_OK):
        if not os.path.exists(PATH + 'snapshots_n/'):
            os.mkdir(PATH + 'snapshots_n/')

        torch.save({
            'initializer': initializer.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'convlstm': convlstm_cell.state_dict(),
            'convlstm_middle': convlstm_middle.state_dict(),
            'convlstm_3': convlstm_3.state_dict(), 
            'optimizer': optimizer.state_dict()
        }, PATH + 'snapshots_n/{}.pth'.format(epoch))


@ex.automain
def main(tr_conf):
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    initializer = Initializer().to(device)
    encoder = EncoderVGGUnet().to(device)
    convlstm_cell = ConvLSTMCell(input_size=(None, None), input_dim=512,
                                 hidden_dim=512, kernel_size=(3, 3), bias=True).to(device)
    decoder = DecoderSkip(tr_conf['num_classes']).to(device)
    convlstm_middle = ConvLSTMCell(input_size=(None, None), input_dim=512,
                                 hidden_dim=512, kernel_size=(3, 3), bias=True).to(device)

    convlstm_3 = ConvLSTMCell(input_size=(None, None), input_dim=256,
                                 hidden_dim=256, kernel_size=(5, 5), bias=True).to(device)

    param_list = list(initializer.parameters()) + list(encoder.parameters()) + list(convlstm_cell.parameters()) + list(
        decoder.parameters()) + list(convlstm_middle.parameters()) + list(convlstm_3.parameters())
    
    optimizer = torch.optim.Adam(param_list, lr=1e-5)
    optimizer.zero_grad()
    scheduler = sched.StepLR(optimizer, step_size=4, gamma=0.99)

    im_res = [256, 448]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr = {'image': transforms.Compose([transforms.Resize(im_res),
                                       transforms.ToTensor(),
                                       normalize]),

          'gt': transforms.Compose([transforms.Resize(im_res)])}

    for epoch in range(tr_conf['starting_epoch'], tr_conf['n_epoch']):
        print(epoch, PATH)
        train_set = dl.YoutubeVOS(mode='train',
                                json_path=tr_conf['meta_train_path'],
                                im_path=tr_conf['im_train_path'],
                                ann_path=tr_conf['ann_train_path'],
                                transform=tr,
                                affine=tr_conf['affine_info'],
                                hflip=tr_conf['hflip'],
                                num_border_pixels=tr_conf['border_pixels'],
                                bin_size=tr_conf['bin_size'],
                                max_len = 5+epoch//10
                                )

        train_loader = DataLoader(train_set, batch_size=tr_conf['b_s']//tr_conf['n_workers'], num_workers=tr_conf['n_workers'],
                                      shuffle=True, pin_memory=True, worker_init_fn=dl._init_fn)
        train(initializer, encoder, decoder, convlstm_cell, convlstm_middle, convlstm_3, optimizer, train_loader, epoch, train_all=False)

        if epoch > 150: scheduler.step()


