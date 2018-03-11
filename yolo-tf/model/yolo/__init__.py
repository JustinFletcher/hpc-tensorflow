"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import configparser
import os
import re
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
from . import inference


def calc_cell_xy(cell_height, cell_width, dtype=np.float32):
    cell_base = np.zeros([cell_height, cell_width, 2], dtype=dtype)
    for y in range(cell_height):
        for x in range(cell_width):
            cell_base[y, x, :] = [x, y]
    return cell_base


class Model(object):
    def __init__(self, net, scope, classes, boxes_per_cell, training=False):
        _, self.cell_height, self.cell_width, _ = tf.get_default_graph().get_tensor_by_name(scope + '/conv:0').get_shape().as_list()
        cells = self.cell_height * self.cell_width
        with tf.name_scope('regress'):
            with tf.name_scope('inputs'):
                end = cells * classes
                self.prob = tf.reshape(net[:, :end], [-1, cells, 1, classes], name='prob')
                inputs_remaining = tf.reshape(net[:, end:], [-1, cells, boxes_per_cell, 5], name='inputs_remaining')
                self.iou = tf.identity(inputs_remaining[:, :, :, 0], name='iou')
                self.offset_xy = tf.identity(inputs_remaining[:, :, :, 1:3], name='offset_xy')
                wh01_sqrt_base = tf.identity(inputs_remaining[:, :, :, 3:], name='wh01_sqrt_base')
            wh01 = tf.square(wh01_sqrt_base, name='wh01')
            wh01_sqrt = tf.abs(wh01_sqrt_base, name='wh01_sqrt')
            self.coords = tf.concat([self.offset_xy, wh01_sqrt], -1, name='coords')
            self.wh = tf.identity(wh01 * [self.cell_width, self.cell_height], name='wh')
            _wh = self.wh / 2
            self.offset_xy_min = tf.identity(self.offset_xy - _wh, name='offset_xy_min')
            self.offset_xy_max = tf.identity(self.offset_xy + _wh, name='offset_xy_max')
            self.areas = tf.reduce_prod(self.wh, -1, name='areas')
        if not training:
            with tf.name_scope('detection'):
                cell_xy = calc_cell_xy(self.cell_height, self.cell_width).reshape([1, cells, 1, 2])
                self.xy = tf.identity(cell_xy + self.offset_xy, name='xy')
                self.xy_min = tf.identity(cell_xy + self.offset_xy_min, name='xy_min')
                self.xy_max = tf.identity(cell_xy + self.offset_xy_max, name='xy_max')
                self.conf = tf.identity(tf.expand_dims(self.iou, -1) * self.prob, name='conf')
        self.inputs = net
        self.classes = classes
        self.boxes_per_cell = boxes_per_cell


class Objectives(dict):
    def __init__(self, model, mask, prob, coords, offset_xy_min, offset_xy_max, areas):
        self.model = model
        with tf.name_scope('true'):
            self.mask = tf.identity(mask, name='mask')
            self.prob = tf.identity(prob, name='prob')
            self.coords = tf.identity(coords, name='coords')
            self.offset_xy_min = tf.identity(offset_xy_min, name='offset_xy_min')
            self.offset_xy_max = tf.identity(offset_xy_max, name='offset_xy_max')
            self.areas = tf.identity(areas, name='areas')
        with tf.name_scope('iou') as name:
            _offset_xy_min = tf.maximum(model.offset_xy_min, self.offset_xy_min, name='_offset_xy_min') 
            _offset_xy_max = tf.minimum(model.offset_xy_max, self.offset_xy_max, name='_offset_xy_max')
            _wh = tf.maximum(_offset_xy_max - _offset_xy_min, 0.0, name='_wh')
            _areas = tf.reduce_prod(_wh, -1, name='_areas')
            areas = tf.maximum(self.areas + model.areas - _areas, 1e-10, name='areas')
            iou = tf.truediv(_areas, areas, name=name)
        with tf.name_scope('mask'):
            best_box_iou = tf.reduce_max(iou, 2, True, name='best_box_iou')
            best_box = tf.to_float(tf.equal(iou, best_box_iou), name='best_box')
            mask_best = tf.identity(self.mask * best_box, name='mask_best')
            mask_normal = tf.identity(1 - mask_best, name='mask_normal')
        with tf.name_scope('dist'):
            iou_dist = tf.square(model.iou - mask_best, name='iou_dist')
            coords_dist = tf.square(model.coords - self.coords, name='coords_dist')
            prob_dist = tf.square(model.prob - self.prob, name='prob_dist')
        with tf.name_scope('objectives'):
            cnt = np.multiply.reduce(iou_dist.get_shape().as_list())
            self['iou_best'] = tf.identity(tf.reduce_sum(mask_best * iou_dist) / cnt, name='iou_best')
            self['iou_normal'] = tf.identity(tf.reduce_sum(mask_normal * iou_dist) / cnt, name='iou_normal')
            self['coords'] = tf.identity(tf.reduce_sum(tf.expand_dims(mask_best, -1) * coords_dist) / cnt, name='coords')
            self['prob'] = tf.identity(tf.reduce_sum(tf.expand_dims(self.mask, -1) * prob_dist) / cnt, name='prob')


class Builder(object):
    def __init__(self, args, config):
        section = __name__.split('.')[-1]
        self.args = args
        self.config = config
        with open(os.path.join(utils.get_cachedir(config), 'names'), 'r') as f:
            self.names = [line.strip() for line in f]
        self.boxes_per_cell = config.getint(section, 'boxes_per_cell')
        self.func = getattr(inference, config.get(section, 'inference'))
    
    def __call__(self, data, training=False):
        _scope, self.output = self.func(data, len(self.names), self.boxes_per_cell, training=training)
        with tf.name_scope(__name__.split('.')[-1]):
            self.model = Model(self.output, _scope, len(self.names), self.boxes_per_cell)
    
    def create_objectives(self, labels):
        section = __name__.split('.')[-1]
        self.objectives = Objectives(self.model, *labels)
        with tf.name_scope('weighted_objectives'):
            for key in self.objectives:
                tf.add_to_collection(tf.GraphKeys.LOSSES, tf.multiply(self.objectives[key], self.config.getfloat(section + '_hparam', key), name='weighted_' + key))
