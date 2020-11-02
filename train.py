import argparse
import torch
from torch.utils import data
from tqdm import tqdm
from models.cvae import *
from loader.train_loader import TrainLoader
from utils import *
from tensorboardX import SummaryWriter
import open3d as o3d
import torch.optim as optim
import itertools
import sys
import chamfer_pytorch.dist_chamfer as ext


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='runs_try', help='path to save train logs and models')
parser.add_argument('--batch_size', type=int, default=120, help='input batch size')
parser.add_argument('--num_workers', type=int, default=2, help='# of dataloadeer num_workers')
parser.add_argument('--lr_h', type=float, default=0.0001, help='learning rate for adam')
parser.add_argument('--num_epoch', type=int, default=300000, help='# of training epochs ')
parser.add_argument("--log_step", default=1000, type=int, help='log after n iters')
parser.add_argument("--save_step", default=2000, type=int, help='save models after n iters')

parser.add_argument('--dataset_path', type=str, default='/mnt/hdd/PROX', help='path to prox dataset')
parser.add_argument('--preprocess_file_path', type=str,
                    default='/mnt/hdd/PROX/preprocessed_encoding',
                    help='path to preprocessed bps features')

parser.add_argument("--weight_loss_kl", default=0.5, type=float, help='loss weight of kl')
parser.add_argument("--weight_loss_contact", default=0.01, type=float, help='loss weight of contact loss')
parser.add_argument("--start_contact_loss", default=200, type=int, help='from which epoch to start backpropogating contact loss')

parser.add_argument("--weight_loss_body_bps", default=1.0, type=float, help='loss weight of body bps feature reconstruction')
parser.add_argument("--weight_loss_scene_bps", default=1.0, type=float, help='loss weight of scene bps feature reconstruction')
parser.add_argument("--weight_loss_scene_bps_verts", default=1.0, type=float, help='loss weight of scene bps verts reconstruction')
parser.add_argument("--weight_loss_body_verts", default=1.0, type=float, help='loss weight of body verts reconstruction')

parser.add_argument("--cube_size", default=2.0, type=float, help='size of 3D cage')
parser.add_argument("--eps_d", default=32, type=int, help='dimension of latent z')
parser.add_argument("--num_bps", default=10000, type=int, help='# of basis points')

parser.add_argument('--load_trained_models', action='store_true', help='load pretrained models')
parser.add_argument('--scene_bps_AE_path', type=str, default='')
parser.add_argument('--cVAE_path', type=str, default='')
parser.add_argument('--scene_verts_AE_path', type=str, default='')
parser.add_argument('--bodyDec_path', type=str, default='')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(writer, logger):
    ################ load preprocessed bps feats / verts #############
    # load training data from separate npy files (into memory)
    # if run compute_bps_encoding.py multiple times for each scene
    # change train_scene_list to the corresponding names defined in compute_bps_encoding.py
    # e.x. ['BasementSittingBooth_1', 'BasementSittingBooth_2', ...]
    train_scene_list = ['BasementSittingBooth',
                        'MPH8',
                        'MPH11',
                        'MPH112',
                        'N0Sofa',
                        'N3Library',
                        'Werkraum',
                        'N3Office',
                        ]
    scene_bps_list_train, body_bps_list_train, body_verts_list_train = [], [], []
    scene_bps_verts_list_train = []

    shift_list_train, rot_angle_list_train = [], []
    train_scene_name_list = []
    print('[INFO] loading training preprocessed data ...')

    cur_n_sample = 0
    for scene_name in tqdm(train_scene_list):
        scene_bps_list_train += list(np.load('{}/{}_scene_bps_list.npy'.format(args.preprocess_file_path, scene_name)))
        body_bps_list_train += list(np.load('{}/{}_body_bps_list.npy'.format(args.preprocess_file_path, scene_name)))
        scene_bps_verts_list_train += list(np.load('{}/{}_scene_bps_verts_local_list.npy'.format(args.preprocess_file_path, scene_name)))
        body_verts_list_train += list(np.load('{}/{}_body_verts_local_list.npy'.format(args.preprocess_file_path, scene_name)))

        shift_list_train += list(np.load('{}/{}_shift_list.npy'.format(args.preprocess_file_path, scene_name)))
        rot_angle_list_train += list(np.load('{}/{}_rot_list.npy'.format(args.preprocess_file_path, scene_name)))
        cur_n_sample = len(shift_list_train) - cur_n_sample
        train_scene_name_list += ([scene_name[0:-2]] * cur_n_sample)
        cur_n_sample = len(shift_list_train)

    print('[INFO] training preprocessed bps/verts loaded.')
    n_sample_train = len(scene_bps_list_train)
    print('[INFO] {} training samples in total.'.format(n_sample_train))

    # load test data
    test_scene_list = ['MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
    scene_bps_list_test, body_bps_list_test, body_verts_list_test = [], [], []
    scene_bps_verts_list_test = []
    print('[INFO] loading test preprocessed data ...')

    for scene_name in tqdm(test_scene_list):
        scene_bps_list_test += list(np.load('{}/{}_scene_bps_list.npy'.format(args.preprocess_file_path, scene_name)))
        body_bps_list_test += list(np.load('{}/{}_body_bps_list.npy'.format(args.preprocess_file_path, scene_name)))
        scene_bps_verts_list_test += list(np.load('{}/{}_scene_bps_verts_local_list.npy'.format(args.preprocess_file_path, scene_name)))
        body_verts_list_test += list(np.load('{}/{}_body_verts_local_list.npy'.format(args.preprocess_file_path, scene_name)))
    print('[INFO] test preprocessed bps/verts loaded.')
    n_sample_test = len(scene_bps_list_test)
    print('[INFO] {} test samples in total.'.format(n_sample_test))

    ####################### read scene vertex (prox coordinate) #######
    scene_list = ['BasementSittingBooth', 'MPH8', 'MPH11', 'MPH112', 'N0Sofa', 'N3Library', 'Werkraum', 'N3Office',
                  'MPH1Library', 'MPH16', 'N0SittingBooth', 'N3OpenArea']
    scene_mesh_path = os.path.join(args.dataset_path, 'scenes_downsampled')
    scene_verts_dict = dict()
    for scene_name in tqdm(scene_list):
        scene_o3d = o3d.io.read_triangle_mesh(os.path.join(scene_mesh_path, scene_name + '.ply'))
        scene_verts_dict[scene_name] = np.asarray(scene_o3d.vertices)


    ######################## set dataloader ########################
    train_dataset = TrainLoader(mode='body_verts_contact')
    train_dataset.n_samples = n_sample_train
    train_dataset.scene_bps_list = scene_bps_list_train    # [n_sample, n_feat, n_bps]
    train_dataset.body_bps_list = body_bps_list_train
    train_dataset.body_verts_list = body_verts_list_train  # [n_sample, n_body_verts, 3]
    train_dataset.scene_bps_verts_list = scene_bps_verts_list_train

    train_dataset.shift_list = shift_list_train
    train_dataset.rotate_list = rot_angle_list_train
    train_dataset.scene_name_list = train_scene_name_list
    train_dataset.scene_verts_dict = scene_verts_dict

    print('[INFO] train dataloader set, select n_samples={}'.format(train_dataset.__len__()))
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, drop_last=True)

    test_dataset = TrainLoader(mode='body_verts')
    test_dataset.n_samples = n_sample_test
    test_dataset.scene_bps_list = scene_bps_list_test  # [n_sample, n_feat, n_bps]
    test_dataset.body_bps_list = body_bps_list_test
    test_dataset.body_verts_list = body_verts_list_test  # [n_sample, n_body_verts, 3]
    test_dataset.scene_bps_verts_list = scene_bps_verts_list_test
    print('[INFO] test dataloader set, select n_samples={}'.format(test_dataset.__len__()))
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=True)

    ######################## set train configs ###########################
    scene_bps_AE = BPSRecMLP(n_bps=args.num_bps, n_bps_feat=1, hsize1=1024,  hsize2=512,).to(device)
    c_VAE = BPS_CVAE(n_bps=args.num_bps, n_bps_feat=1, hsize1=1024,  hsize2=512, eps_d=args.eps_d).to(device)

    scene_verts_AE = Verts_AE(n_bps=args.num_bps, hsize1=1024, hsize2=512).to(device)
    body_dec = Body_Dec_shift(n_bps=args.num_bps, n_bps_feat=1, hsize1=1024, hsize2=512, n_body_verts=10475,
                              body_param_dim=75, rec_goal='body_verts').to(device)

    if args.load_trained_models:
        weights = torch.load(args.scene_bps_AE_path, map_location=lambda storage, loc: storage)
        scene_bps_AE.load_state_dict(weights)
        weights = torch.load(args.cVAE_path, map_location=lambda storage, loc: storage)
        c_VAE.load_state_dict(weights)
        weights = torch.load(args.scene_verts_AE_path, map_location=lambda storage, loc: storage)
        scene_verts_AE.load_state_dict(weights)
        weights = torch.load(args.bodyDec_path, map_location=lambda storage, loc: storage)
        body_dec.load_state_dict(weights)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  itertools.chain(scene_bps_AE.parameters(),
                                                  c_VAE.parameters(),
                                                  scene_verts_AE.parameters(),
                                                  body_dec.parameters())),
                           lr=args.lr_h)

    # body parts to compute contact loss
    contact_part = ['L_Leg', 'R_Leg']
    vid, _ = get_contact_id(body_segments_folder=os.path.join(args.dataset_path, 'body_segments'),
                            contact_body_parts=contact_part)

    ####################### train #########################################
    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            scene_bps_AE.train()
            c_VAE.train()
            scene_verts_AE.train()
            body_dec.train()

            total_steps += 1
            [body_verts, scene_bps, body_bps, scene_bps_verts, shift, rotate, scene_verts] = [item.to(device) for item in data]

            optimizer.zero_grad()

            body_verts = body_verts / args.cube_size
            scene_bps_verts = scene_bps_verts / args.cube_size

            # body bps cvae
            scene_bps_rec, scene_bps_feat = scene_bps_AE(scene_bps)  # [bs, n_bps_feat, n_bps], [bs, n_bps_feat, hsize2]
            body_bps_rec, mu, logvar = c_VAE(body_bps, scene_bps_feat)
            loss_rec_scene_bps = F.l1_loss(scene_bps, scene_bps_rec)
            loss_rec_body_bps = F.l1_loss(body_bps, body_bps_rec)
            loss_kl = 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
            loss_kl = torch.sqrt(loss_kl * loss_kl + 1)

            # body regressor
            scene_bps_verts_rec, scene_bps_verts_feat = scene_verts_AE(scene_bps_verts)
            body_verts_rec, body_shift = body_dec(body_bps_rec, scene_bps_verts_feat)  # [bs, 3, 10475], [bs, 3]
            loss_rec_scene_bps_verts = F.l1_loss(scene_bps_verts, scene_bps_verts_rec)
            # shift generated body
            body_shift = body_shift.repeat(1, 1, 10475).reshape([body_verts_rec.shape[0], 10475, 3])  # [bs, 10475, 3]
            body_verts_rec_shift = body_verts_rec + body_shift.permute(0, 2, 1)  # [bs, 3, 10475]
            loss_rec_body_all = F.l1_loss(body_verts, body_verts_rec)

            # start contact loss
            loss_contact = torch.tensor(0.0)
            if epoch >= args.start_contact_loss:
                # convert body_verts_rec_shift to prox coordinate, shift: [bs, 3], rotate: [bs, 1]
                bs = body_verts_rec_shift.shape[0]
                body_verts_rec_shift_prox = torch.zeros([bs, 10475, 3]).to(device)  # [bs, 10475, 3]
                shift = shift.repeat(1, 1, 10475).reshape([bs, 10475, 3])  # [bs, 10475, 3]
                rotate = rotate.repeat(1, 10475).reshape([bs, 10475])  # [bs, 10475]
                temp = body_verts_rec_shift.permute(0, 2, 1) * args.cube_size - shift  # recover to scale before unit ball scaling
                body_verts_rec_shift_prox[:, :, 0] = temp[:, :, 0] * torch.cos(-rotate) - temp[:, :, 1] * torch.sin(-rotate)
                body_verts_rec_shift_prox[:, :, 1] = temp[:, :, 0] * torch.sin(-rotate) + temp[:, :, 1] * torch.cos(-rotate)
                body_verts_rec_shift_prox[:, :, 2] = temp[:, :, 2]

                # contact loss
                body_verts_contact = body_verts_rec_shift_prox[:, vid, :]  # [bs,1121,3]
                dist_chamfer_contact = ext.chamferDist()
                # scene_verts: [bs, 50000, 3]
                contact_dist, _ = dist_chamfer_contact(body_verts_contact.contiguous(), scene_verts.contiguous())
                loss_contact = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0))


            loss = args.weight_loss_kl * loss_kl + \
                   args.weight_loss_body_bps * loss_rec_body_bps + args.weight_loss_scene_bps * loss_rec_scene_bps + \
                   args.weight_loss_scene_bps_verts * loss_rec_scene_bps_verts + \
                   args.weight_loss_body_verts * loss_rec_body_all + \
                   args.weight_loss_contact * loss_contact
            loss.backward()
            optimizer.step()

            if total_steps % args.log_step == 0:
                # cvae
                writer.add_scalar('train/loss_rec_scene_bps', loss_rec_scene_bps.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_scene_bps: {:.6f}'. \
                    format(step, epoch, loss_rec_scene_bps.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_rec_body_bps', loss_rec_body_bps.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_bps: {:.6f}'. \
                    format(step, epoch, loss_rec_body_bps.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_kl', loss_kl.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_kl: {:.6f}'. \
                    format(step, epoch, loss_kl.item())
                logger.info(print_str)
                print(print_str)

                # body regressor
                writer.add_scalar('train/loss_rec_scene_bps_verts', loss_rec_scene_bps_verts.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_scene_bps_verts: {:.6f}'. \
                    format(step, epoch, loss_rec_scene_bps_verts.item())
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_rec_body_verts_all', loss_rec_body_all.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_verts_all: {:.6f}'. \
                    format(step, epoch, loss_rec_body_all.item())
                logger.info(print_str)
                print(print_str)

                # contact loss
                writer.add_scalar('train/loss_contact', loss_contact.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_contact: {:.6f}'. \
                    format(step, epoch, loss_contact.item())
                logger.info(print_str)
                print(print_str)


            ################## test loss #################################
            if total_steps % args.log_step == 0:
                loss_rec_scene_bps_test, loss_rec_body_bps_test = 0, 0
                loss_rec_scene_bps_verts_test, loss_rec_body_all_test = 0, 0
                with torch.no_grad():
                    for test_step, data in tqdm(enumerate(test_dataloader)):
                        scene_bps_AE.eval()
                        c_VAE.eval()
                        scene_verts_AE.eval()
                        body_dec.eval()
                        [body_verts_test, scene_bps_test, body_bps_test, scene_bps_verts_test] = [item.to(device) for item in data]
                        body_verts_test = body_verts_test / args.cube_size
                        scene_bps_verts_test = scene_bps_verts_test / args.cube_size

                        scene_bps_rec_test, scene_bps_feat_test = scene_bps_AE(scene_bps_test)
                        body_bps_rec_test, _, _ = c_VAE(body_bps_test, scene_bps_feat_test)
                        loss_rec_scene_bps_test += F.l1_loss(scene_bps_test, scene_bps_rec_test).item()
                        loss_rec_body_bps_test += F.l1_loss(body_bps_test, body_bps_rec_test).item()

                        scene_bps_verts_rec_test, scene_bps_verts_feat_test = scene_verts_AE(scene_bps_verts_test)
                        body_verts_rec_test, body_shift_test = body_dec(body_bps_rec_test, scene_bps_verts_feat_test)
                        loss_rec_scene_bps_verts_test += F.l1_loss(scene_bps_verts_test, scene_bps_verts_rec_test).item()
                        loss_rec_body_all_test += F.l1_loss(body_verts_test, body_verts_rec_test).item()


                loss_rec_scene_bps_test = loss_rec_scene_bps_test / test_step
                loss_rec_body_bps_test = loss_rec_body_bps_test / test_step
                loss_rec_scene_bps_verts_test = loss_rec_scene_bps_verts_test / test_step
                loss_rec_body_all_test = loss_rec_body_all_test / test_step

                writer.add_scalar('test/loss_rec_scene_bps', loss_rec_scene_bps_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_scene_bps_test: {:.6f}'. \
                    format(step, epoch, loss_rec_scene_bps_test)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_rec_body_bps', loss_rec_body_bps_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_bps_test: {:.6f}'. \
                    format(step, epoch, loss_rec_body_bps_test)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_rec_scene_bps_verts', loss_rec_scene_bps_verts_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_scene_bps_verts_test: {:.6f}'. \
                    format(step, epoch, loss_rec_scene_bps_verts_test)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('test/loss_rec_body_verts_all', loss_rec_body_all_test, total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_rec_body_verts_all_test: {:.6f}'. \
                    format(step, epoch, loss_rec_body_all_test)
                logger.info(print_str)
                print(print_str)



            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "sceneBpsAE_last_model.pkl")
                torch.save(scene_bps_AE.state_dict(), save_path)
                save_path = os.path.join(writer.file_writer.get_logdir(), "cVAE_last_model.pkl")
                torch.save(c_VAE.state_dict(), save_path)
                save_path = os.path.join(writer.file_writer.get_logdir(), "sceneBpsVertsAE_last_model.pkl")
                torch.save(scene_verts_AE.state_dict(), save_path)
                save_path = os.path.join(writer.file_writer.get_logdir(), "body_dec_last_model.pkl")
                torch.save(body_dec.state_dict(), save_path)
                print('[*] last model saved\n')
                logger.info('[*] last model saved\n')



if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)




