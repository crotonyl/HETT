import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms

# from r2r.agent_cmt import Seq2SeqCMTAgent
from multiagent.actions import Action
from multiagent.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from multiagent.models.dark_net import Darknet
from multiagent.models.CLIP import CLIP
# from direction.models.ddppo.resenet_encoders import TorchVisionResNet50
from multiagent.models.goal_predictor import GoalPredictor, MapEncoder
from multiagent.observation import cropclient
from multiagent.space import Pose4D, Point2D, Point3D
from multiagent.teacher.algorithm.lookahead import lookahead_discrete_action
from multiagent.teacher.trajectory import _moved_pose
from models.vln_model import CustomBERTModel
from models.ET_haa import ET
from transformers import AutoModel, BertTokenizerFast
# import clip
import cv2
import shapely
import shapely.geometry
from shapely.geometry import Polygon, MultiPoint
from logger import write_to_record_file, print_progress, timeSince


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    tensors = tensors.items()
    for line in tensors:
        print('{}\t{}'.format(*line))


# https://programmerah.com/using-shapely-geometry-polygon-to-calculate-the-iou-of-any-two-quadrilaterals-28395/
def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def is_default_gpu(opts) -> bool:
    return opts.local_rank == -1 or dist.get_rank() == 0


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0:  # lng is postive
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


class NavCMTAgent:
    def __init__(self, args, allow_ngpus=True, rank=0):
        self.results = {}
        self.losses = []  # For learning agents
        self.args = args
        self.env = []
        self.env_name = ''
        random.seed(1)

        # RGB normalization values
        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models

        self.tokenizer = BertTokenizerFast.from_pretrained('/cver/xcding/code/tokenizer_files/bert-base-uncase')
        self.lang_model = CustomBERTModel().cuda()

        # self.img_tensor = transforms.ToTensor()

        self.vision_model = Darknet(self.args.darknet_model_file, 224).cuda()

        new_state = torch.load(self.args.darknet_weight_file)
        state = self.vision_model.state_dict()
        model_keys = set(state.keys())
        state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
        state.update(state_dict)
        self.vision_model.load_state_dict(state)



        # create the et model
        self.vln_model = ET(self.args).cuda()
        # self.map_encoder = MapEncoder(240)
        # self.goal_predictpr = GoalPredictor(240, 7)
        self.progress_regression = nn.MSELoss(reduction='sum')

        if self.args.world_size > 1 and allow_ngpus:
            self.lang_model = DDP(self.lang_model, broadcast_buffers=False, find_unused_parameters=True,
                                  device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.vision_model = DDP(self.vision_model, broadcast_buffers=False, find_unused_parameters=True,
                                    device_ids=[self.args.local_rank], output_device=self.args.local_rank)
            self.vln_model = DDP(self.vln_model, broadcast_buffers=False, find_unused_parameters=True,
                                 device_ids=[self.args.local_rank], output_device=self.args.local_rank)

            # self.lang_model = nn.DataParallel(self.lang_model).cuda()
            # self.vision_model = nn.DataParallel(self.vision_model).cuda()
            # self.vln_model = nn.DataParallel(self.vln_model).cuda()
            self.lang_model_without_ddp = self.lang_model.module
            self.vision_model_without_ddp = self.vision_model.module
            self.vln_model_without_ddp = self.vln_model.module


        else:
            self.lang_model_without_ddp = self.lang_model
            self.vision_model_without_ddp = self.vision_model
            self.vln_model_without_ddp = self.vln_model

        # self.vln_model = ViT_LSTM(
        #     self.args, 
        #     self.vision_model).cuda()

        # optimizer        
        assert args.optim in ("adam", "adamW")
        OptimizerClass = torch.optim.Adam if args.optim == "adam" else torch.optim.AdamW
        self.et_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vln_model.parameters()),
                                           lr=args.learning_rate)
        self.lang_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.lang_model.parameters()),
                                                   lr=self.args.learning_rate)
        self.vision_model_optimizer = OptimizerClass(filter(lambda p: p.requires_grad, self.vision_model.parameters()),
                                                     lr=self.args.learning_rate)
        self.optimizers = (self.et_optimizer, self.lang_model_optimizer, self.vision_model_optimizer)
        # self.optimizers = (self.et_optimizer, self.lang_model_optimizer)

        #         # Optimizers
        # if self.args.optim == 'rms':
        #     optimizer = torch.optim.RMSprop
        # elif self.args.optim == 'adam':
        #     optimizer = torch.optim.Adam
        # elif self.args.optim == 'adamW':
        #     optimizer = torch.optim.AdamW
        # elif self.args.optim == 'sgd':
        #     optimizer = torch.optim.SGD
        # else:
        #     assert False
        # if self.default_gpu:
        #     print('Optimizer: %s' % self.args.optim)

        # self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        # self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        # self.lang_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.lang_model.parameters()), lr=self.args.lr)
        # self.vln_model_optimizer = optimizer(filter(lambda p: p.requires_grad, self.vln_model.parameters()), lr=self.args.lr)
        # self.optimizers = (self.lang_model_optimizer, self.vln_model_optimizer)

        # Evaluations
        self.losses = []
        self.progress_regression = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def get_results(self):

        return self.results

    def visualize(self, loader, env_name='no_name_provided', feedback='student', not_in_train=False, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        self.lang_model.eval()
        self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        idx = 0
        start = time.time()
        for l in loader:
            idx += 1
            for traj in self.rollout(visualize=True):  # loop for #batch times
                self.loss = 0
                self.results[traj['instr_id']] = traj
                # print(traj)
            tot = loader.dataset.size() / self.env.batch_size
            print_progress(idx, tot,
                           prefix='Progress:', suffix='%s (%d/%d)' % (
                    timeSince(start, float(idx) / tot), idx, tot), bar_length=80)


    def test(self, loader, env_name='no_name_provided', feedback='student', not_in_train=False, **kwargs):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        self.env_name = env_name

        self.vln_model.eval()
        self.lang_model.eval()
        self.vision_model.eval()

        self.losses = []
        self.results = {}
        self.loss = 0
        idx = 0
        start = time.time()
        for l in loader:
            idx += 1
            for traj in self.rollout(visualize=False):  # loop for #batch times
                self.loss = 0
                self.results[traj['instr_id']] = traj
                # print(traj)
            tot = loader.dataset.size() / self.env.batch_size
            print_progress(idx, tot,
                           prefix='Progress:', suffix='%s (%d/%d)' % (
                    timeSince(start, float(idx) / tot), idx, tot), bar_length=80)

    def train(self, loader, n_epochs, feedback='student', nss_w_weighting=1, **kwargs):
        ''' Train for a given number of epochs '''
        self.feedback = feedback

        self.lang_model.train()
        self.vln_model.train()
        self.vision_model.train()

        self.losses = []
        for epoch in range(1, n_epochs + 1):
            idx = 0
            start = time.time()
            # print('?')
            for _, l in enumerate(loader):
                idx += 1
                # if idx >= 100:
                #     break
                # train_loop_start_time = time.time()
                self.lang_model_optimizer.zero_grad()
                self.vision_model_optimizer.zero_grad()
                self.et_optimizer.zero_grad()
                self.loss = 0

                if feedback == 'teacher':
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.teacher_weight)
                elif feedback == 'student':  # agents in teacher and student separately

                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight)  # self.args.nss_w*nss_w_weighting, **kwargs)
                    # if epoch_train > 10000:
                    self.feedback = 'student'
                    self.rollout(train_ml=self.args.ml_weight)
                else:
                    assert False

                # print("--- One rollout takes %s seconds ---" % (time.time() - train_loop_start_time))

                # print(self.rank, epoch, self.loss)
                # torch.autograd.set_detect_anomaly(True)

                self.loss.backward()
                # print('suc')

                torch.nn.utils.clip_grad_norm_(self.vln_model.parameters(), 40.)

                self.lang_model_optimizer.step()
                self.vision_model_optimizer.step()
                self.et_optimizer.step()
                # print("---------- One iter takes %s seconds ---" % (time.time() - train_loop_start_time))

                if self.default_gpu:
                    tot = n_epochs * loader.dataset.size() / self.env.batch_size
                    # print('is')
                    print_progress(idx, tot,
                                   prefix='Progress:', suffix='%s (%d/%d)' % (
                            timeSince(start, float(idx) / tot), idx, tot), bar_length=80)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def rollout(self, train_ml=None, visualize=False):

        # rollout_start_time = time.time()

        obs = self.env._get_obs(random_direction=(self.feedback == 'teacher'))
        batch_size = len(obs)

        # Language input
        lang_inputs = []
        for i, ob in enumerate(obs):
            # if self.args.vision_only:
            #     lang_inputs.append('')
            # else:
            lang_inputs.append(ob['instruction'])
        encoding = self.tokenizer(lang_inputs, padding=True, return_tensors="pt")
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        lang_features, linear_cls, cls_hidden = self.lang_model(input_ids, attention_mask)

        # lang_features --> 768
        # linear_cls --> 49 (used to attend to img features)
        # c_0 = cls_hidden

        # print(lang_features.size()) # batch_size*sequence_length*768

        # Record starting points of the current batch
        current_directions = [np.array(ob['pose'].yaw, dtype=np.float32) for ob in obs]
        current_positions = [np.array(ob['position'], dtype=np.float32) for ob in obs]
        poses = [ob['pose'] for ob in obs]
        direction_t = torch.from_numpy(np.array(current_directions, dtype=np.float32))
        position_t = torch.from_numpy(np.array(current_positions, dtype=np.float32))
        traj = [defaultdict(list) for ob in obs]

        global_position = np.stack([np.array([i, j], dtype=np.float32)
                                    for i in range(self.args.grid_size)
                                    for j in range(self.args.grid_size)]) / self.args.grid_size

        global_positions = torch.from_numpy(np.stack([global_position
                                                      for _ in range(batch_size)
                                                      ])).cuda()

        for i, ob in enumerate(obs):
            traj[i]['goal'] = ob['goal']
            traj[i]['instr_id'] = ob['id']
            # rounds = lang_inputs[i].split('[QUE]')
            # remove = 0
            # for r in rounds:
            #     if 'Yes' in r[0:5]:
            #         remove += 1
            # traj[i]['num_dia'] = len(rounds) - remove
            # traj[i]['path_corners'] = [(np.array(ob['gt_path_corners'][0]), ob['starting_angle'])]
            traj[i]['gt_trajectory'] = ob['trajectory']
            traj[i]['trajectory'] = [poses[i]]
            traj[i]['stage1_trajectory'] = [poses[i]]
        # print(np.array([len(ob['trajectory']) for ob in obs]))

        # Initialization the finishing status
        ended = np.array([False] * batch_size)

        # Init the logs
        # ml_loss = 0.
        direction_loss = torch.tensor(0.).cuda()
        progress_loss = 0.
        goal_predict_loss = 0.
        target_predict_loss = 0.

        stage1_step = 0
        stage2_step = 0
        stage2_rotate = 0

        input = {
            'directions': torch.zeros((batch_size, 0, 4)).cuda(),
            'grid_fts': torch.zeros(batch_size, 0, 768).cuda(),
            'grid_index': torch.zeros(batch_size, 0).cuda(),
            'frames': torch.zeros(batch_size, 0, 512, 49).cuda(),
            'lenths': [0 for _ in range(batch_size)],
            'lang': lang_features,
            'candidates': global_positions,
            'centroids': torch.zeros((batch_size, 0, 2)).cuda(),
            'lang_cls': linear_cls,
            'map_fts': torch.zeros(batch_size, 0, 512, 49).cuda(),
        }

        stage1_ended = np.array([False] * batch_size)

        for t in range(self.args.max_action_len):

            # print("- action rollingout takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # rollingout_action_start_time = time.time()
            images = []
            for i in range(len(obs)):
                images.append(obs[i]['rgb'].copy())
            images = np.stack(images)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # W x H x C to C x W x H
            images = np.ascontiguousarray(images, dtype=np.float32)
            images -= self.rgb_mean
            images /= self.rgb_std
            im_feature = self.vision_model(torch.from_numpy(images).cuda())
            im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)






            current_direct = direction_t.view(-1, 1).cuda()
            current_pos = position_t.view(-1, 2).cuda()
            direction = torch.concat(
                (torch.sin(current_direct), torch.cos(current_direct), current_pos), axis=1)

            # if self.args.no_direction:
            #     input['directions'] = torch.hstack((input['directions'], torch.zeros_like(direction.view(-1, 1, 2))))
            # else:
            input['directions'] = direction.view(-1, 1, 4)
            # if self.args.language_only:
            #     input['frames'] = torch.hstack((input['frames'], torch.zeros_like(im_feature.view(-1, 1, 512, 49))))
            # else:
            # print(input['frames'].shape, im_feature.shape)
            input['frames'] = im_feature.view(-1, 1, 512, 49)
            input['maps'] = torch.from_numpy(np.array([ob['maps'] for ob in obs], dtype=np.float32)).cuda()


            centroid_lens = np.array(len(ob['centroids']) for ob in obs)
            input['centroids'] = torch.from_numpy(np.array([ob['centroids'] for ob in obs], dtype=np.float32)).cuda()
            # input['directions'] = direction.view(-1,1,2)
            # input['frames'] = im_feature.view(-1,1, 512,49)

            for i in range(len(obs)):
                if not ended[i]:
                    input['lenths'][i] += 1

            # print('.')





            pred_direction, pred_progress, pred_goals, pred_logits, grid_ft = self.vln_model(
                directions=input['directions'],
                frames=input['frames'],
                lenths=input['lenths'],
                grid_fts=input['grid_fts'],
                grid_index=input['grid_index'],
                maps=input['maps'],
                lang=input['lang'],
                candidates=input['candidates'],
                centroids=input['centroids'],
                lang_cls=input['lang_cls']
            )

            input['grid_fts'] = torch.cat((input['grid_fts'], grid_ft), dim=1)
            grid_index = torch.tensor(np.array([ob['cur_grid'] for ob in obs])).unsqueeze(1).cuda()
            # print(input['grid_index'], grid_index)

            input['grid_index'] = torch.cat((input['grid_index'], grid_index), dim=1)
            # print("- model prediction takes %s seconds ---" % (time.time() - rollingout_action_start_time))
            # pred_direction = output
            # pred_progress = progress

            # Predicted progress
            pred_progress_t = pred_progress.cpu().detach().numpy()

            # Predicted waypoint
            nt_direct = torch.atan2(pred_direction[:, 0], pred_direction[:, 1])
            at_direction = nt_direct.cpu().detach().numpy()
            # for i in range(len(a_t_next_pos_ratio)):
            #     max_of_a_t_next_pos_i = max(abs(a_t_next_pos_ratio[i][0]), abs(a_t_next_pos_ratio[i][1]), 1)
            #     a_t_next_pos_ratio[i][0] /= max_of_a_t_next_pos_i
            #     a_t_next_pos_ratio[i][1] /= max_of_a_t_next_pos_i

            # Predicted altitude
            # a_t_altitude = pred_altitude.cpu().detach().numpy()
            #
            # # Clip the prediction to (0,1)
            # for i in range(len(a_t_altitude)):
            #     a_t_altitude[i] = min(1., max(0., a_t_altitude[i]))
            # for i in range(len(pred_progress_t)):
            #     pred_progress_t[i] = min(1., max(0., pred_progress_t[i]))
            gt_direction = np.array([ob['direction'] for ob in obs], dtype=np.float32)
            gt_goal = torch.from_numpy(np.array([ob['normalized_goal'] for ob in obs], dtype=np.float32))
            gt_progress = torch.from_numpy(np.array([ob['progress'] for ob in obs], dtype=np.float32))
            gt_target = torch.from_numpy(np.array([ob['grid_goal'] for ob in obs], dtype=np.int64))
            # there is no ground truth in unseen_test set
            if not 'test' in self.env_name:
                # Get ground truth
                # print(t, target, gt_progress)

                # Compute loss

                for i in range(len(obs)):
                    true_direction = torch.tensor(gt_direction[i])

                    true_sin = torch.sin(true_direction)
                    true_cos = torch.cos(true_direction)
                    true_sin_cos = torch.stack([true_sin, true_cos], dim=-1).cuda()
                    # gt_progress = torch.tensor(obs[i]['progress']).cuda()
                    # print(pred_direction[i].view(-1).shape, pred_progress[i].view(-1).shape, true_sin_cos.shape, gt_progress[i].view(-1).shape)
                    # cuda_gt_next_pos_ratio = torch.from_numpy(target[i][0]).cuda()
                    # print(pred_direction[i].view(-1), true_sin_cos)
                    if not ended[i]:
                        # if stage1_ended[i]:
                        direction_loss += self.progress_regression(pred_direction[i].view(-1), true_sin_cos)

                        progress_loss += self.progress_regression(pred_progress[i].view(-1),
                                                                  gt_progress[i].view(-1).cuda())
                        goal_predict_loss += F.mse_loss(pred_goals[i].view(-1), gt_goal[i].view(-1).cuda())
                        # print(pred_goals[i], gt_goal[i], goal_predict_loss)

                    # ml_loss += direction_loss
                    # ml_loss += progress_loss

                    # print(ml_loss)
                    if direction_loss != direction_loss:  # debug for nan loss
                        print('0', direction_loss)
                    if progress_loss != progress_loss:  # debug for nan loss
                        print('0', progress_loss)
                # print(at_direction, gt_direction, ml_loss)
                target_predict_loss += self.criterion(pred_logits, gt_target.unsqueeze(1).cuda())
                # print(pred_logits.shape)
            # Log the trajectory
            # print(at_direction, gt_direction)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['actions'].append(at_direction[i])
                    if not 'test' in self.env_name:
                        traj[i]['gt_actions'].append(gt_direction[i])
                        traj[i]['gt_progress'].append(gt_progress[i].item())
                        traj[i]['gt_goal'].append(gt_goal[i])
                    traj[i]['progress'].append(pred_progress[i].item())

            if self.feedback == 'teacher':
                at_goal = gt_goal
                # print('teacher', at_goal.shape)
                a_t = gt_direction
                pred_progress_t = gt_progress
            elif self.feedback == 'student':  # student
                a_t = at_direction
                at_goal = pred_goals

                # _, at_goal = pred_logits.max(1)
                # at_goal = at_goal.squeeze(1)
                # at_goal = gt_goal
                # print('student', at_goal.shape)
            else:
                sys.exit('Invalid feedback option')

            cpu_goal = at_goal.cpu().detach().numpy()
            # print(cpu_goal)

            # Interact with the simulator with actions
            for i in range(len(obs)):

                dst = self.env.unnormalize_position(cpu_goal[i], obs[i]['map_name'],
                                                    self.args.map_meters)

                # gt_center = self.env.unnormalize_position(global_position[gt_goal.cpu().detach().numpy()[i]], obs[i]['map_name'],
                #                                     self.args.map_meters)
                # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])
                if ended[i]:
                    continue
                # if dst.dist_to(poses[i].xy) < 10:
                #     ended[i] = True
                #     continue


                elif pred_progress_t[i] > 0.95 and self.feedback == 'student' and stage1_ended[i]:
                    # Updated 'ended' list and make environment action
                    ended[i] = True
                    continue
                elif t == self.args.max_action_len:
                    ended[i] = True
                    continue

                # print(cpu_goal[i], global_position[cpu_goal[i]])
                # dst = self.env.unnormalize_position(global_position[cpu_goal[i]], obs[i]['map_name'],
                #                                     self.args.map_meters)
                # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])

                # if pred_progress_t[i] < 0.75 and dst.dist_to(poses[i].xy) > 20 and not stage1_ended[i]:
                # if pred_progress_t[i] < 0.75 and dst.dist_to(poses[i].xy) > 10 and not stage1_ended[i]:

                # if pred_progress_t[i] > 0.9 and not stage1_ended[i]:
                #     stage1_ended[i] = True
                if dst.dist_to(poses[i].xy) > 5 and not stage1_ended[i]:
                    stage1_step += 1
                    traj[i]['pred_goal'].append(dst)
                    # pred_goal_xys = [
                    #     unnormalize_position(global_position[goal_id] / args.grid_size, eps.map_name, args.map_meters)
                    #     for eps, goal_id in zip(episodes_batch, goal_ids)]
                    # dst = Point2D(obs[i]['centroid_goal'][0], obs[i]['centroid_goal'][1])
                    # dst = self.env.unnormalize_position(global_position[cpu_goal[i]], obs[i]['map_name'], self.args.map_meters)
                    if self.feedback == 'teacher':
                        cur_step = stage1_step * self.args.move_iteration
                        cur_step = cur_step if cur_step < len(obs[i]['trajectory']) else -1
                        poses[i] = obs[i]['trajectory'][cur_step]
                    else:
                        poses[i] = self.move(poses[i], dst,
                                         self.args.move_iteration)
                    if not ended[i]:
                        traj[i]['stage1_trajectory'].append(poses[i])

                elif abs(a_t[i]) < np.pi / 12:
                    stage1_ended[i] = True
                    stage2_step += 1
                    poses[i] = _moved_pose(poses[i], *Action(5, 0, 0))
                    if len(traj[i]['stage2_trajectory']) == 0:
                        traj[i]['stage2_trajectory'].append(traj[i]['stage1_trajectory'][-1])
                    if not ended[i]:
                        traj[i]['stage2_trajectory'].append(poses[i])
                else:
                    stage1_ended[i] = True
                    stage2_rotate += 1
                    poses[i] = _moved_pose(poses[i], *Action(0, a_t[i], 0))
                    if len(traj[i]['stage2_trajectory']) == 0:
                        traj[i]['stage2_trajectory'].append(traj[i]['stage1_trajectory'][-1])
                    if not ended[i]:
                        traj[i]['stage2_trajectory'].append(poses[i])

            # Save trajectory output
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(poses[i])
                    # Update the status
            direction_t = torch.from_numpy(np.array(current_directions, dtype=np.float32))
            position_t = torch.from_numpy(np.array(current_positions, dtype=np.float32))
            obs = self.env._get_obs(poses, random_direction=(self.feedback == 'teacher'))  # get gt_obs
            # current_view_corners = [np.array(ob['gt_path_corners'][0]) for ob in obs]

            # Early exit if all ended
            if ended.all():
                break
        # print(visualize)
        if visualize:
            for i, ob in enumerate(obs):
                print('visualization/%s_%d_%d' % (
                                                ob['id'][0], ob['id'][1], ob['id'][2]))
                dist = traj[i]['trajectory'][-1].xy.dist_to(traj[i]['gt_trajectory'][-1].xy)
                # print(dist)
                # print(dist)
                # if dist > 20:
                #     continue
                landmarks = self.env.nav_maps[i].landmark_map.get_contours()
                ladnmark_names = self.env.nav_maps[i].landmark_map.landmark_names
                # print(self.env.split)
                # print(obs[i]['grid_goal'])
                # print(ob['id'])
                cropclient.save_grids(ob['id'][0], traj[i]['stage1_trajectory'], traj[i]['stage2_trajectory'],
                                     traj[i]['gt_trajectory'], landmarks, ladnmark_names,
                                     os.path.join(GOAL_PREDICTOR_CHECKPOINT_DIR,
                                                  'visualization/%s_%d_%d' % (
                                                ob['id'][0], ob['id'][1], ob['id'][2])),
                                     traj[i]['pred_goal'])
        if train_ml is not None:
            # print(ml_loss)
            # ml_loss = direction_loss + progress_loss
            ml_loss = 1 * direction_loss + 0.1 * progress_loss + 2 * goal_predict_loss + 0.1 * target_predict_loss
            # ml_loss = progress_loss + goal_predict_loss
            self.loss += ml_loss * train_ml / batch_size

            # self.logs['ml_loss'].append((ml_loss * train_ml / batch_size).item())

            self.logs['direction_loss'].append((direction_loss * train_ml / batch_size).item())
            self.logs['progress_loss'].append((progress_loss * train_ml / batch_size).item())
            self.logs['goal_predict_loss'].append((goal_predict_loss * train_ml / batch_size).item())
            self.logs['target_predict_loss'].append((target_predict_loss * train_ml / batch_size).item())
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        # if t==0:
        #     self.logs
        self.logs['stage1_step'].append(float(stage1_step) / batch_size)
        self.logs['stage2_step'].append(float(stage2_step) / batch_size)
        self.logs['stage2_rotate'].append(float(stage2_rotate) / batch_size)

        # print('[3]')
        # debug_memory()
        # print()
        return traj



    def move(self, pose: Pose4D, dst: Point2D, iterations: int):
        dst = Point3D(dst.x, dst.y, pose.z)

        for _ in range(iterations):
            action = lookahead_discrete_action(pose, [dst])
            pose = _moved_pose(pose, *action.value)
            # if(action.name == 'STOP'):

        return pose

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("lang_model", self.lang_model_without_ddp, self.lang_model_optimizer),
                     ("vision_model", self.vision_model_without_ddp, self.vision_model_optimizer),
                     ("vln_model", self.vln_model_without_ddp, self.et_optimizer),
                     ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys == load_keys:
                print("NOTICE: LOADing ALL KEYS IN THE ", name)
                state_dict = states[name]['state_dict']
            else:
                print("NOTICE: DIFFERENT KEYS IN THE ", name)
                # if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = {k: v for k, v in states[name]['state_dict'].items() if k in model_keys}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])

            def count_parameters(mo):
                return sum(p.numel() for p in mo.parameters() if p.requires_grad)

            print('Model parameters: ', count_parameters(model))

        all_tuple = [("lang_model", self.lang_model_without_ddp, self.lang_model_optimizer),
                     ("vision_model", self.vision_model_without_ddp, self.vision_model_optimizer),
                     ("vln_model", self.vln_model_without_ddp, self.et_optimizer),
                     ]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_model']['epoch'] - 1
