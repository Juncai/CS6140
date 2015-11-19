import DataLoader as loader
import numpy as np
import random
import os.path
import math
import generate_thresholds as gt


def random_select_data(tr_save_path, sel_tr_save_path, percent):
    all_tr = loader.load_pickle_file(tr_save_path)

    tr_l_ind_dict = {}
    selected_tr_data = [[], []]
    for i in range(10):
        tr_l_ind_dict[i] = [l_ind for l_ind, l in enumerate(all_tr[1]) if l == i]
    for i in range(10):
        i_n = len(tr_l_ind_dict[i])
        pick_n = int(percent * i_n)
        cur_pick_ind = np.random.choice(tr_l_ind_dict[i], pick_n, replace=False).tolist()
        selected_tr_data[0].extend([x for x_ind, x in enumerate(all_tr[0]) if x_ind in cur_pick_ind])
        selected_tr_data[1].extend([y for y_ind, y in enumerate(all_tr[1]) if y_ind in cur_pick_ind])
    loader.save(sel_tr_save_path, selected_tr_data)

def abstract_features(data_path, cs_path, rects_path, res_path):
    # get cs
    cs = get_cs(data_path, cs_path)
    rects = loader.load_pickle_file(rects_path)

    # 2 features for each rectangle
    features = []
    for i, ccs in enumerate(cs):
        f = []
        for rect in rects:
            f.extend(compute_feature_with_cs(rect, ccs))
        features.append(f)
        print('{} rects finished.'.format(i))

    # combine with labels
    label = loader.load_pickle_file(data_path)[1]
    f_l = [np.array(features), label]
    loader.save(res_path, f_l)

    return f_l


def get_cs(data_path, cs_path):
    # dp compute cheat sheet
    cs = None
    if os.path.isfile(cs_path):
        cs = loader.load_pickle_file(cs_path)
        print('CS loaded.')
    else:
        print('Start compute cs.')
        data = loader.load_pickle_file(data_path)
        cs = dp_compute_cs(data[0])
        loader.save(cs_path, cs)
        print('CS saved.')
    return cs

def compute_feature_with_cs(rect, cs):
    '''

    :param rect: (upper left, lower right)
    :param cs:
    :return:
    '''
    # since the coordinates in the rect is inclusive
    # a0 | a1
    # -------
    # a2 | a3
    mid_y = math.floor((rect[0][1] + rect[1][1]) / 2)
    mid_x = math.floor((rect[0][0] + rect[1][0]) / 2)

    a01_black = count_black((rect[0], (mid_x, rect[1][1])), cs)
    a23_black = count_black(((mid_x + 1, rect[0][1]), rect[1]), cs)
    a02_black = count_black((rect[0], (rect[1][0], mid_y)), cs)
    a13_black = count_black(((rect[0][0], mid_y + 1), rect[1]), cs)

    return [a02_black - a13_black, a01_black - a23_black]


def count_black(rect, cs):
    #  ul     up
    # left | rect
    a_all = cs[rect[1][0]][rect[1][1]]
    a_left = cs[rect[1][0]][rect[0][1] - 1] if rect[0][1] > 0 else 0
    a_up = cs[rect[0][0] - 1][rect[1][1]] if rect[0][0] > 0 else 0
    a_ul = cs[rect[0][0] - 1][rect[0][1] - 1] if rect[0][0] > 0 and rect[0][1] > 0 else 0
    return a_all - a_left - a_up + a_ul


def dp_compute_cs(images):
    css = []
    for i, img in enumerate(images):
        css.append(dp_compute_cs_helper(img))
        print('{} rects finished.'.format(i))
    return css

def dp_compute_cs_helper(image):
    cur_cs = np.zeros(np.shape(image))
    cur_cs[0][0] = 1 if image[0][0] > 0 else 0
    for i in range(1, len(image)):
        cur_cs[i][0] = cur_cs[i-1][0] + (1 if image[i][0] > 0 else 0)
    for i in range(1, len(image[0])):
        cur_cs[0][i] = cur_cs[0][i-1] + (1 if image[0][i] > 0 else 0)
    for i in range(1, len(image)):
        for j in range(1, len(image[0])):
            cur_cs[i][j] = cur_cs[i][j-1] + cur_cs[i-1][j] - cur_cs[i-1][j-1] + (1 if image[i][j] > 0 else 0)
    return cur_cs

def random_select_rectangle(h, w, n, pl, ph, save_path=None):
    '''

    :param h: height of the image in pixel
    :param w: width of the image in pixel
    :param n: number of rectangle
    :param pl: min pixels of each rectangle
    :param ph: max pixels of each rectangle
    :return:
    '''
    sel_rects = []
    for i in range(n):
        a = -1
        while a < pl or a > ph:
            p1 = (random.randint(0, h - 1), random.randint(0, w - 1))
            p2 = (random.randint(0, h - 1), random.randint(0, w - 1))
            a = rect_area(p1, p2)
        sel_rects.append(((min(p1[0], p2[0]), min(p1[1], p2[1])), (max(p1[0], p2[0]), max(p1[1], p2[1]))))

    if save_path is not None:
        loader.save(save_path, sel_rects)

    return sel_rects

def rect_area(p1, p2):
    return (abs(p2[0] - p1[0]) + 1) * (abs(p2[1] - p1[1]) + 1)


def convert_to_np_array(path):
    data = loader.load_pickle_file(path)
    # convert labels
    np_label = np.array(data[1])
    np_features = np.array(data[0])
    loader.save(path, [np_features, np_label])


if __name__ == '__main__':
    tr_save_path = 'data\\digits\\tr_data.pickle'
    te_save_path = 'data\\digits\\te_data.pickle'
    te_cs_save_path = 'data\\digits\\te_cs.pickle'
    sel_tr_save_path = 'data\\digits\\sel_tr_data.pickle'
    sel_tr_cs_save_path = 'data\\digits\\sel_tr_cs.pickle'
    rects_path = 'data\\digits\\100_rects.pickle'
    tr_f_l_path = 'data\\digits\\tr_f_l.pickle'
    te_f_l_path = 'data\\digits\\te_f_l.pickle'
    thresh_path = 'data\\digits\\sel_tr.threshes'
    percent = 0.2

    # randomly pick 20 percent of the training data
    random_select_data(tr_save_path, sel_tr_save_path, percent)
    convert_to_np_array(sel_tr_save_path)

    # pre compute the cheatsheet
    get_cs(sel_tr_save_path, sel_tr_cs_save_path)
    get_cs(te_save_path, te_cs_save_path)

    # randomly pick 100 rectangles
    random_select_rectangle(28, 28, 100, 140, 170, rects_path)
    abstract_features(sel_tr_save_path, sel_tr_cs_save_path, rects_path, tr_f_l_path)
    abstract_features(te_save_path, te_cs_save_path, rects_path, te_f_l_path)

    # generate thresholds
    data = loader.load_pickle_file(tr_f_l_path)
    gt.generate_thresholds(data[0], thresh_path)

    tmp = loader.load_pickle_file(tr_f_l_path)
    print('done')
