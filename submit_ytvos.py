import cv2, json, os
import numpy as np
import scipy.misc as sm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
import dataloader as dl
from networks import *
from sacred import Experiment
from sacred.observers import FileStorageObserver


ex = Experiment()

# where checkpoints are located 
base_path = '../snapshots_n/' 

# where the sacred experiment will locate
PATH = base_path + 'submission_info/'
ex.observers.append(FileStorageObserver.create(PATH))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@ex.config
def config():
    model_num = None
    decoder_sig = None
    tr_conf = {
        'model_dir': base_path + model_num,
        'submission_dir': PATH + 'SUBMISSION_DIR_{}/'.format(model_num),
        'scores_dir': PATH + 'SCORES/',
        'rgb_dir': '/ds2/YoutubeVOS2018/valid/JPEGImages/',
        'ann_dir': '/ds2/YoutubeVOS2018/valid/Annotations/',
        'meta_dir': '/ds2/YoutubeVOS2018/valid/meta.json',
        'test_all': False,
        'apply_sigmoid': not decoder_sig,
    }


convlstm_3 = ConvLSTMCell(input_size=(None, None), input_dim=256,
                                 hidden_dim=256, kernel_size=(5, 5), bias=True).to(device)

initializer = Initializer().to(device)
encoder = EncoderVGGUnet().to(device)
convlstm_cell = ConvLSTMCell(input_size=(12, 25), input_dim=512,
                                 hidden_dim=512, kernel_size=(3, 3), bias=True).to(device)
decoder = DecoderSkip(42).to(device)

convlstm_middle = ConvLSTMCell(input_size=(12, 25), input_dim=512,
                                 hidden_dim=512, kernel_size=(3, 3), bias=True).to(device)


class Submission:
    def __init__(self, conf):

        self.apply_sigmoid = conf['apply_sigmoid']
        self.model_dir = conf['model_dir']
        self.submission_dir = conf['submission_dir']
        if not os.path.exists(self.submission_dir):
            os.mkdir(self.submission_dir)

        self.results_dir = conf['scores_dir']
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.rgb_dir = conf['rgb_dir']
        self.ann_dir = conf['ann_dir']

        with open(conf['meta_dir'], 'r') as f:
            data = f.read()
        self.meta = json.loads(data)

        self.test_all = conf['test_all']
        self.palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                        64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

        self.conf = conf

    def save_score_maps(self, test_loader):

        # load the model 
        c_p = torch.load(self.model_dir)
        initializer.load_state_dict(c_p['initializer'])
        encoder.load_state_dict(c_p['encoder'])
        decoder.load_state_dict(c_p['decoder'])
        convlstm_cell.load_state_dict(c_p['convlstm'])
        convlstm_middle.load_state_dict(c_p['convlstm_middle'])
        convlstm_3.load_state_dict(c_p['convlstm_3'])

        initializer.eval()
        encoder.eval()
        decoder.eval()
        convlstm_cell.eval()
        convlstm_middle.eval()
        convlstm_3.eval()

        with torch.no_grad():
            for sequence in test_loader:
                seq_name = sequence['seq_name'][0]
                categories = list(sequence.keys())[1:]

                if not os.path.exists(self.results_dir + seq_name):
                    os.makedirs(self.results_dir + seq_name)

                save_path = self.results_dir + seq_name + '/'
                for cat in categories:
                    rgb, mask, names = sequence[cat]['image'], sequence[cat]['first_mask'], sequence[cat]['name']
                    # save score map of gt 
                    temp = os.path.splitext(names[0][0])[0]
                    np.save(save_path + temp + '_instance_%02d.npy' % int(cat), mask.squeeze().cpu().numpy())

                    init_frame = torch.cat([rgb[0].to(device), mask.to(device)], dim=1)
                    states = initializer(init_frame)
                    h, c = states[0]
                    h_middle, c_middle = states[1]
                    h_3, c_3 = states[2]

                    for ii in range(1, len(rgb)):
                        enc, mead_feats = encoder(rgb[ii].to(device))
                        h, c = convlstm_cell(enc, (h, c))
                        h_middle, c_middle = convlstm_middle(mead_feats[3], (h_middle, c_middle))
                        h_3, c_3 = convlstm_3(mead_feats[2], (h_3, c_3))

                        decoded_img, distance_scores = decoder(h, mead_feats, h_middle, h_3)
                        temp = os.path.splitext(names[ii][0])[0]

                        if self.apply_sigmoid: decoded_img = torch.sigmoid(decoded_img)
                        np.save(save_path + temp + '_instance_%02d.npy' % int(cat), decoded_img.squeeze().cpu().numpy())

    def merge_score_maps(self, test_loader):
        with torch.no_grad():
            for sequence in test_loader:
                seq_name = sequence['seq_name'][0]

                mask_path = os.path.join(self.submission_dir, seq_name)
                if not os.path.exists(mask_path):
                    os.mkdir(mask_path)

                frames = sorted(os.listdir(self.rgb_dir + seq_name))
                score_maps = sorted(os.listdir(self.results_dir + seq_name))

                for f in frames:
                    f_score_list = []
                    f_ids = []
                    for sm in score_maps:
                        if sm.startswith(f[:5]):
                            sm_path = os.path.join(self.results_dir, seq_name, sm)
                            # map & id
                            f_score_list.append(np.load(sm_path))
                            f_ids.append(int(sm[-6:-4]))

                    if len(f_ids) == 0:
                        print('EMPTY FRAME FOUND', f, seq_name)

                    obj_ids_ext = np.array([0] + f_ids, dtype=np.uint8)
                    bg_score = np.ones((256, 448)) * 0.5

                    scores = [bg_score] + f_score_list
                    scores_all = np.stack(scores, axis=0)
                    pred_idx = scores_all.argmax(axis=0)
                    label_pred = obj_ids_ext[pred_idx]

                    res_im = Image.fromarray(label_pred, mode='P')
                    res_im.putpalette(self.palette)

                    res_im.save(os.path.join(mask_path, f[:5] + '.png'))


@ex.automain
def main(tr_conf):

    im_res = [256, 448]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr = {'image': transforms.Compose([transforms.Resize(im_res),
                                       transforms.ToTensor(),
                                       normalize]),

          'gt': transforms.Compose([transforms.Resize(im_res)])}

    test_set = dl.YoutubeVOS(mode='test',
                             json_path=tr_conf['meta_dir'],
                             im_path=tr_conf['rgb_dir'],
                             ann_path=tr_conf['ann_dir'],
                             transform=tr)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, pin_memory=True)

    sub = Submission(tr_conf)
    sub.save_score_maps(test_loader)
    sub.merge_score_maps(test_loader)
