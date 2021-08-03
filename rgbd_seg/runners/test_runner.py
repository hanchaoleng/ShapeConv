import os

import cv2
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F

from .inference_runner import InferenceRunner
from ..utils import gather_tensor, vis_utils


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        extra_data = len(self.test_dataloader.dataset) % self.world_size
        self.test_exclude_num = self.world_size - extra_data if extra_data != 0 else 0
        self.tta = test_cfg.get('tta', False)
        self.save_pred = test_cfg.get('save_pred', False)
        if self.save_pred:
            self.classes = test_cfg['data']['dataset']['classes']
            self.dir_save_pred = os.path.join(base_cfg['workdir'], 'pred_vis')
            if not os.path.isdir(self.dir_save_pred):
                os.makedirs(self.dir_save_pred)
            self.logger.info("pred image save at " + self.dir_save_pred)
            self.dir_save_label = os.path.join(base_cfg['workdir'], 'label_vis')
            if not os.path.isdir(self.dir_save_label):
                os.makedirs(self.dir_save_label)
            self.logger.info("label image save at " + self.dir_save_label)

    def __call__(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start testing')
        with torch.no_grad():
            img_id = 0
            for idx, (image, mask) in enumerate(self.test_dataloader):
                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                if self.tta:
                    output = self._tta_compute(image)
                else:
                    output = self.model(image)
                    output = self.compute(output)

                output = gather_tensor(output)
                mask = gather_tensor(mask)
                if self.save_pred:
                    preds = output.cpu().numpy()
                    labels = mask.cpu().numpy()
                    for pred, label in zip(preds, labels):
                        # if img_id == 15:
                        #     print(label)
                        pred_rgb = vis_utils.visualize_seg(pred, vis_utils.get_color_map(self.classes),
                                                           self.classes)[0] * 255
                        pred_rgb[label == 255] = np.array((0, 0, 0))
                        path_pred_out = os.path.join(self.dir_save_pred, "%d.png" % (img_id + 1))
                        img = Image.fromarray(pred_rgb.astype(np.uint8))
                        img.save(path_pred_out)

                        label_rgb = vis_utils.visualize_seg(label, vis_utils.get_color_map(self.classes),
                                                            self.classes)[0] * 255
                        path_label_out = os.path.join(self.dir_save_label, "%d.png" % (img_id + 1))
                        img = Image.fromarray(label_rgb.astype(np.uint8))
                        img.save(path_label_out)

                        img_id += 1
                    # break

                if idx + 1 == len(self.test_dataloader) and self.test_exclude_num > 0:
                    output = output[:-self.test_exclude_num]
                    mask = mask[:-self.test_exclude_num]

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.accumulate()
                self.logger.info('Test, Iter {}, {}'.format(
                    idx + 1,
                    ', '.join(['{}: {}'.format(k, np.round(v, 4)) for k, v in
                               res.items()])))

        self.logger.info('Test Result: {}'.format(', '.join(
            ['{}: {}'.format(k, np.round(v, 4)) for k, v in res.items()])))

        return res

    def _tta_compute(self, image):
        b, c, h, w = image.size()
        probs = []
        for scale, bias in zip(self.tta['scales'], self.tta['biases']):
            if bias:
                new_h, new_w = int(h * scale + bias), int(w * scale + bias)
            else:
                bias_h = h * scale % 16
                if bias_h > 8:
                    new_h = int(h * scale + (16 - bias_h))
                else:
                    new_h = int(h * scale - bias_h)
                bias_w = w * scale % 16
                if bias_w > 8:
                    new_w = int(w * scale + (16 - bias_w))
                else:
                    new_w = int(w * scale - bias_w)
            new_img = F.interpolate(image, size=(new_h, new_w),
                                    mode='bilinear', align_corners=True)
            output = self.model(new_img)
            probs.append(output)

            if self.tta['flip']:
                flip_img = new_img.flip(3)
                flip_output = self.model(flip_img)
                prob = flip_output.flip(3)
                probs.append(prob)

        if self.multi_label:
            prob = torch.stack(probs, dim=0).sigmoid().mean(dim=0)
            prob = torch.where(prob >= 0.5,
                               torch.full_like(prob, 1),
                               torch.full_like(prob, 0)).long()  # b c h w
        else:
            prob = torch.stack(probs, dim=0).softmax(dim=2).mean(dim=0)
            _, prob = torch.max(prob, dim=1)  # b h w
        return prob
