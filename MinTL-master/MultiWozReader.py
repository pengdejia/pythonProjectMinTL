# !usr/bin/env python 
# -*- coding:utf-8 _*-
"""
@Author:Derrick
 
@File:MultiWozReader.py
@Time:2022/2/19 14:49
 
"""

from collections import OrderedDict
import torch
from collections import Counter
import numpy as np
from itertools import chain
from copy import deepcopy
import os, random, csv, logging, json

from damd_multiwoz import ontology
from damd_multiwoz.db_ops import MultiWozDB
from damd_multiwoz.config import global_config_v2 as cfg

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def set_cfg_batch_size(batch_size=32):
    cfg.batch_size = batch_size


class _ReaderBase(object):
    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            # print(f"cfg batch_size{cfg.batch_size}")
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder < 1/10 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        if len(batch) > 0.1 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_batches(self, set_name):
        global dia_count
        # log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            # log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n"%(
            #         k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            all_batches += batches
        # log_str += 'total batch num: %d\n'%len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches
        # logging.info(log_str)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

    def save_result(self, write_mode, results, field, write_title=False):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title + '\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None

    def save_result_report(self, results):
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv' % cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s' % str(cfg.beam_width)
            if cfg.beam_diverse_param > 0:
                setting += ', penalty=%s' % str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s' % str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s' % str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn': cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
               'decode': cfg.aspn_decode_mode, 'param': setting, 'nbest': cfg.nbest,
               'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'],
               'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class MultiWozReader(_ReaderBase):
    def __init__(self, vocab=None, args=None):
        super().__init__()
        self.db = MultiWozDB(cfg.dbs)
        self.args = args
        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(open(cfg.slot_value_set_path, 'r').read())
        test_list = [l.strip().lower() for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(cfg.dev_list, 'r').readlines()]
        logger.info("test data path{} dev data path{}".format(cfg.test_list, cfg.dev_list))
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        self.vocab = vocab
        self.vocab_size = vocab.vocab_size

        self._load_data()
        print("load data finished")

    def _load_data(self, save_temp=False):
        logger.info("train data path{}".format(cfg.data_path + cfg.data_file))
        self.data = json.loads(open(cfg.data_path + cfg.data_file, 'r', encoding='utf-8').read().lower())
        self.train, self.dev, self.test = [], [], []

        # TODO add scheme information for guide the information generate
        scheme_information_ids = None

        """
            way1 
        """
        # scheme_information_list = []
        # for domain in ontology.all_domains:
        #     scheme_information_list.append("[" + domain + "]")
        # for slot in ontology.all_slots:
        #     scheme_information_list.append(slot)

        """
        way2
        """
        # scheme_information_list = []
        # for domain_slot in ontology.all_domain_slots:
        #     domain, slot = domain_slot.split("-")
        #     scheme_information_list.append("[" + domain + "]")
        #     scheme_information_list.append(slot)
        #
        # scheme_information_ids = self.vocab.tokenizer.encode(
        #     " ".join(scheme_information_list)) + self.vocab.tokenizer.encode('<eos_s>')
        #
        # logger.info("add scheme information ids is {}.".format(len(scheme_information_ids)))

        for fn, dial in self.data.items():
            if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                if self.dev_files.get(fn):
                    self.dev.append(self._get_encoded_data(fn, dial, scheme_information_ids))
                elif self.test_files.get(fn):
                    self.test.append(self._get_encoded_data(fn, dial, scheme_information_ids))
                else:
                    self.train.append(self._get_encoded_data(fn, dial, scheme_information_ids))
        logger.info("train num {}, dev num {} , test num {}".format(len(self.train), len(self.dev), len(self.test)))
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def _get_encoded_data(self, fn, dial, scheme_information_ids=None):
        encoded_dial = []
        dial_context = []
        # delete_op = self.vocab.tokenizer.encode("<None>")  #[32157]

        prev_constraint_dict = {}
        for idx, t in enumerate(dial['log']):
            enc = {}
            enc['dial_id'] = fn
            dial_context.append(self.vocab.tokenizer.encode(t['user']) + self.vocab.tokenizer.encode('<eos_u>'))
            enc['resp_nodelex'] = self.vocab.tokenizer.encode(t['resp_nodelex']) + self.vocab.tokenizer.encode(
                '<eos_r>')
            enc['user'] = list(
                chain(*dial_context[-self.args.context_window:]))  # here we use user to represent dialogue history

            # print(len(enc['user']))
            if scheme_information_ids is not None:
                enc['user'] = scheme_information_ids + enc['user']
            # print(len(enc['user']))

            enc['bspn'] = self.vocab.tokenizer.encode(t['constraint']) + self.vocab.tokenizer.encode('<eos_b>')
            constraint_dict = self.bspan_to_constraint_dict(t['constraint'])

            # TODO add multilabel classify for the project
            update_bspn, update_dict = self.check_update(prev_constraint_dict, constraint_dict)
            enc['domain_slot_list'] = self.get_domain_slot(update_dict)

            enc['update_bspn'] = self.vocab.tokenizer.encode(update_bspn)
            encoded_dial.append(enc)
            prev_constraint_dict = constraint_dict
            dial_context.append(enc['resp_nodelex'])
        return encoded_dial

    def get_domain_slot(self, update_dict):
        """
        # {'hotel': {'pricerange': 'cheap', 'type': 'hotel'}}
        # print("update_dict", update_dict)
        # print(ontology.all_domain_slots)
        #
        :param update_dict:
        :return:
        """
        result_list = [0] * len(ontology.domain_slot_id_dict)
        if len(update_dict) == 0:
            return result_list
        for domain, slot_value_dict in update_dict.items():
            for slot, value in slot_value_dict.items():
                if domain + "-" + slot in ontology.domain_slot_id_dict:
                    index = ontology.domain_slot_id_dict[domain + "-" + slot]
                    result_list[index] = 1
        return result_list

    def check_update(self, prev_constraint_dict, constraint_dict):
        update_dict = {}
        if prev_constraint_dict == constraint_dict:
            return '<eos_b>', update_dict
        for domain in constraint_dict:
            if domain in prev_constraint_dict:
                for slot in constraint_dict[domain]:
                    if constraint_dict[domain].get(slot) != prev_constraint_dict[domain].get(slot):
                        if domain not in update_dict:
                            update_dict[domain] = {}
                        update_dict[domain][slot] = constraint_dict[domain].get(slot)
                # if delete is needed
                # if len(prev_constraint_dict[domain])>len(constraint_dict[domain]):
                for slot in prev_constraint_dict[domain]:
                    if constraint_dict[domain].get(slot) is None:
                        update_dict[domain][slot] = "<None>"
            else:
                update_dict[domain] = deepcopy(constraint_dict[domain])

        update_bspn = self.constraint_dict_to_bspan(update_dict)
        return update_bspn, update_dict

    def constraint_dict_to_bspan(self, constraint_dict):
        if not constraint_dict:
            return "<eos_b>"
        update_bspn = ""
        for domain in constraint_dict:
            if len(update_bspn) == 0:
                update_bspn += f"[{domain}]"
            else:
                update_bspn += f" [{domain}]"
            for slot in constraint_dict[domain]:
                update_bspn += f" {slot} {constraint_dict[domain][slot]}"
        update_bspn += f" <eos_b>"
        return update_bspn

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        # add decoded(str) here
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == "[slot]":
                continue
            if cons == '<eos_b>':
                break

            # deal with domain information
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]

            # deal with slot information
            elif cons in ontology.get_slot:
                # without domain slot is useless
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx + 1]
                        ns = self.vocab.decode(ns) if type(ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains

    def convert_batch(self, batch, prev, first_turn=False, dst_start_token=0):
        """
        user: dialogue history ['user']
        input: previous dialogue state + dialogue history
        TODO: add special token for guide the generate
        output1: dialogue state update ['update_bspn'] or current dialogue state ['bspn']
        """
        inputs = {}
        pad_token = self.vocab.tokenizer.encode("<pad>")[0]
        batch_size = len(batch['user'])
        # input: previous dialogue state + dialogue history
        input_ids = []

        # TODO add the domain slot
        domain_slot_labels = []
        for i in range(batch_size):
            domain_slot_labels.append(batch['domain_slot_list'][i])

        if first_turn:
            for i in range(batch_size):
                input_ids.append(self.vocab.tokenizer.encode('<eos_b>') + batch['user'][i])
        else:
            for i in range(batch_size):
                input_ids.append(prev['bspn'][i] + batch['user'][i])
        input_ids, masks = self.padInput(input_ids, pad_token)
        inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        inputs["masks"] = torch.tensor(masks, dtype=torch.long)

        inputs["domain_slots"] = torch.tensor(domain_slot_labels, dtype=torch.long)

        if self.args.noupdate_dst:
            # here we use state_update denote the belief span (bspn)...
            state_update, state_input = self.padOutput(batch['bspn'], pad_token)
        else:
            state_update, state_input = self.padOutput(batch['update_bspn'], pad_token)
        inputs["state_update"] = torch.tensor(state_update, dtype=torch.long)  # batch_size, seq_len
        inputs["state_input"] = torch.tensor(
            np.concatenate((np.ones((batch_size, 1)) * dst_start_token, state_input[:, :-1]), axis=1), dtype=torch.long)
        # for k in inputs:
        #     if k=="masks":
        #         print(k)
        #         print(inputs[k])
        #     else:
        #         print(k)
        #         print(inputs[k].tolist())
        #         print(k)
        #         print(self.vocab.tokenizer.decode(inputs[k].tolist()[0]))

        return inputs

    def padOutput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        output_ids = np.ones((num_samples, max_len)) * (-100)  # -100 ignore by cross entropy
        decoder_inputs = np.ones((num_samples, max_len)) * pad_token
        for idx, s in enumerate(sequences):
            trunc = s[:max_len]
            output_ids[idx, :lengths[idx]] = trunc
            decoder_inputs[idx, :lengths[idx]] = trunc
        return output_ids, decoder_inputs

    def padInput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        input_ids = np.ones((num_samples, max_len)) * pad_token
        masks = np.zeros((num_samples, max_len))

        for idx, s in enumerate(sequences):
            trunc = s[-max_len:]
            input_ids[idx, :lengths[idx]] = trunc
            masks[idx, :lengths[idx]] = 1
        return input_ids, masks

    def update_bspn(self, prev_bspn, bspn_update, domain_slot_list=ontology.all_domain_slots):
        constraint_dict_update = self.bspan_to_constraint_dict(self.vocab.tokenizer.decode(bspn_update))
        if not constraint_dict_update:
            return prev_bspn
        constraint_dict = self.bspan_to_constraint_dict(self.vocab.tokenizer.decode(prev_bspn))
        # print(constraint_dict_update)
        # print(domain_slot_list)
        # print(constraint_dict)
        for domain in constraint_dict_update:
            if domain not in constraint_dict:
                constraint_dict[domain] = {}
            for slot, value in constraint_dict_update[domain].items():
                if domain + "-" + slot not in domain_slot_list:
                    continue
                if value == "<None>":  # delete the slot
                    _ = constraint_dict[domain].pop(slot, None)
                else:
                    constraint_dict[domain][slot] = value
        updated_bspn = self.vocab.tokenizer.encode(self.constraint_dict_to_bspan(constraint_dict))
        return updated_bspn

    def wrap_result(self, result_dict, eos_syntax=None):
        decode_fn = self.vocab.sentence_decode
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax

        field = ['dial_id', 'turn_num', 'user', 'bspn_gen', 'bspn']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            # customize for the eval, always skip the first turn, so we create a dummy
            for prop in field[2:]:
                entry[prop] = ''
            results.append(entry)
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)
                    entry[key] = decode_fn(v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        return results, field

    def get_domain_slot_list(self, labels):
        # ontology.all_domain_slots
        predict_domain_slot_lists = list()
        if len(labels.shape) == 2:
            for i in range(labels.shape[0]):
                predict_domain_slot_list = list()
                for j in range(labels.shape[1]):
                    if labels[i][j] == 1:
                        predict_domain_slot_list.append(ontology.all_domain_slots[j])
                predict_domain_slot_lists.append(predict_domain_slot_list)
        else:
            predict_domain_slot_list = list()
            for i in range(labels.shape[0]):
                if labels[i] == 1:
                    predict_domain_slot_list.append(ontology.all_domain_slots[i])
            predict_domain_slot_lists.append(predict_domain_slot_list)
        return predict_domain_slot_lists




