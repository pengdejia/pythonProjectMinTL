import os, random, argparse, time, logging, json
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from itertools import chain
from copy import deepcopy

from model_dst import T5_DST, T5_DST_v2, BART_DST,BartTokenizer

from damd_multiwoz.config import global_config_v2 as cfg
# from utils import _ReaderBase, set_cfg_batch_size
# from damd_multiwoz import ontology
# from damd_multiwoz.db_ops import MultiWozDB

from MultiWozReader import MultiWozReader, set_cfg_batch_size

from damd_multiwoz.eval import MultiWozEvaluator

from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)

from vocab import Vocab


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)





class Model(object):
    def __init__(self, args, test=False, v1=False):
        if args.back_bone=="t5":  
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            if v1:
                self.model = T5_DST.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            else:
                self.model = T5_DST_v2(args=args, test=test)
                if test:
                    self.model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))

        elif args.back_bone=="bart":
            self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = BART_DST.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        vocab = Vocab(self.model, self.tokenizer)

        if not args.is_reader_store:
            self.reader = MultiWozReader(vocab,args)
            torch.save(self.reader, os.path.join(args.data_dir, "data.bin"))
            logger.info("store the reader into path {}".format(args.data_dir) )
        else:
            self.reader = torch.load(os.path.join(args.data_dir, "data.bin"))
            logger.info("load the reader from the path {}".format(args.data_dir))

        logger.info("data load")
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        self.model.to(args.device)
        self.v1 = v1

    def load_model(self):
        if self.args.back_bone=="t5":
            if self.v1:
                self.model = T5_DST.from_pretrained(self.args.model_path)
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, "pytorch_model.bin")))
        elif self.args.back_bone=="bart":
            self.model = BART_DST.from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def train(self):
        btm = time.time()
        step = 0
        prev_min_jg = 0
        print(f"vocab_size:{self.model.T5.config.vocab_size}")
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        self.model.T5.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.tokenizer.save_pretrained(self.args.model_path)
        # self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()
        # jg = self.eval()
        # lr scheduler
        lr_lambda = lambda epoch: self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)
        do_test = True
        for epoch in range(cfg.epoch_num):
            log_dst = 0
            log_cnt = 0
            sw = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {'pv_bspn': None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.T5.config.decoder_start_token_id)

                    for k in inputs:
                        inputs[k] = inputs[k].to(self.args.device)
                        # print(inputs[k].shape)
                    if self.args.is_ablation:
                        dst_loss, labels_loss = self.model(input_ids=inputs["input_ids"],
                                                           attention_mask=inputs["masks"],
                                                           decoder_input_ids=inputs["state_input"],
                                                           lm_labels=inputs["state_update"],
                                                           labels_tensor=None
                                                           )
                    else:
                        dst_loss, labels_loss = self.model(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["masks"],
                                            decoder_input_ids=inputs["state_input"],
                                            lm_labels=inputs["state_update"],
                                            labels_tensor=inputs["domain_slots"]
                                            )
                    # print(dst_loss, labels_loss)
                    if labels_loss is None:
                        dst_loss =dst_loss
                    else:
                        dst_loss = dst_loss + labels_loss

                    py_prev['bspn'] = turn_batch['bspn']

                    total_loss = dst_loss / self.args.gradient_accumulation_steps

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step+=1
                    log_dst +=float(dst_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logger.info(
                            'iter:{} [bspn] loss: {:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                                        log_dst/(log_cnt+ 1e-8),
                                                                                        time.time()-btm,
                                                                                        turn_num+1))
                if epoch > cfg.mini_epoch_num and (iter_num + 1) % (cfg.report_dev) == 0:
                    # jg = self.validate(do_test=do_test)
                    jg = self.eval()
                    logger.info(
                        'epoch: %d,dst test jg: %.3f, total time: %.1fmin' % (epoch + 1,
                                                                        jg,
                                                                        (time.time() - sw) / 60))

            epoch_sup_loss = log_dst/(log_cnt+ 1e-8)

            # jg = self.validate(do_test=do_test)
            jg = self.eval()
            logger.info('epoch: %d, train loss: %.3f, dst test jg: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    jg, (time.time()-sw)/60))

            if jg > prev_min_jg:
                early_stop_count = cfg.early_stop_count
                prev_min_jg = jg
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "pytorch_model.bin"))
                logger.info('best model until now, Model saved')
                #self.save_model(epoch)
            else:
                early_stop_count -= 1
                scheduler.step()
                logger.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))

            if early_stop_count <= 0:
                logger.info("after epoch {}, dev result no increase stop train".format(cfg.early_stop_count))
                break

        self.load_model()
        print('result preview...')
        logger.info(str(cfg.eval_load_path))
        self.eval()

    def validate(self, data='dev', do_test=False):
        self.model.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        multi_label_count = 0
        multi_label_true_list = []

        for batch_num, dial_batch in enumerate(data_iterator):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.T5.config.decoder_start_token_id)
                for k in inputs:
                    inputs[k] = inputs[k].to(self.args.device)
                dst_outputs, single_label_count, single_label_true_list = self.model.inference(tokenizer=self.tokenizer,
                                                                                               reader=self.reader,
                                                                                               prev=py_prev,
                                                                                               input_ids=inputs[
                                                                                                   'input_ids'],
                                                                                               attention_mask=inputs[
                                                                                                   "masks"],
                                                                                               labels_tensor=inputs["domain_slots"]
                                                                                               )
                multi_label_count += single_label_count
                multi_label_true_list.extend(single_label_true_list)

                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, field = self.reader.wrap_result(result_collection)
        if len(multi_label_true_list) != 0:
            jg, slot_f1, slot_acc, slot_cnt, slot_corr, total_turn = self.evaluator.dialog_state_tracking_eval(results, bspn_mode='bspn', multi_label_true_list=multi_label_true_list)
        else:
            jg, slot_f1, slot_acc, slot_cnt, slot_corr, total_turn = self.evaluator.dialog_state_tracking_eval(results,
                                                                                                               bspn_mode='bspn')
        logging.info('validation DST join goal: %2.3f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        logging.info('validation multi label Classification accuracy: {}'.format(multi_label_count/(total_turn+1e-10) * 100))
        self.model.train()
        if do_test:
            print('result preview...')
            self.eval()
        return jg

    def eval(self, data='test'):
        logger.info("start testing ...................................................................")
        self.model.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        multi_label_count = 0
        multi_label_true_list = []

        for batch_num, dial_batch in tqdm(enumerate(data_iterator)):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in tqdm(enumerate(dial_batch), leave=False):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.T5.config.decoder_start_token_id)
                for k in inputs:
                    inputs[k] = inputs[k].to(self.args.device)
                dst_outputs, single_label_count, single_label_true_list = self.model.inference(tokenizer=self.tokenizer,
                                                                                               reader=self.reader,
                                                                                               prev=py_prev,
                                                                                               input_ids=inputs['input_ids'],
                                                                                               attention_mask=inputs["masks"],
                                                                                               labels_tensor=inputs["domain_slots"])
                multi_label_count += single_label_count
                multi_label_true_list.extend(single_label_true_list)
                # print("generate", predict_labels_tensor)
                # print("golden", inputs["domain_slots"])
                # if predict_labels_tensor is not None:
                #     for i in range(predict_labels_tensor.shape[0]):
                #         if inputs["domain_slots"][i, :].equal(predict_labels_tensor[i, :]):
                #             multi_label_true_list.append(1)
                #             multi_label_count += 1
                #         else:
                #             multi_label_true_list.append(0)

                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs
                # print(turn_batch)

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        logger.info("finished testing .................................................................")
        results, field = self.reader.wrap_result(result_collection)
        if len(multi_label_true_list) != 0:
            jg, slot_f1, slot_acc, slot_cnt, slot_corr, total_turn = self.evaluator.dialog_state_tracking_eval(results, bspn_mode='bspn', multi_label_true_list=multi_label_true_list)
        else:
            jg, slot_f1, slot_acc, slot_cnt, slot_corr, total_turn = self.evaluator.dialog_state_tracking_eval(results,
                                                                                                               bspn_mode='bspn')

        logger.info('test DST join goal: %2.3f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        logger.info('test multi label Classification number {} accuracy: {}'.format(multi_label_count,multi_label_count/(total_turn+1e-10) * 100))
        print(len(multi_label_true_list), total_turn)

        # with open(os.path.join(self.args.model_path, 'result.txt'), 'w') as f:
        #     f.write('test DST join goal: %2.3f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        # self.reader.metric_record(metric_results)
        self.model.train()
        return jg

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        # print('total trainable params: %d' % param_cnt)
        return param_cnt


def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments_DST'):
        os.mkdir('./experiments_DST')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="test")
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--gpu_id", type=int, default=3,
                        help="Gpus id")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for train")
    parser.add_argument("--seed", type=int, default=11,
                        help="set fixed seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="./pretrain/t5-base", help="Path, url or short name of the model")
    parser.add_argument("--model_path", type=str, default="./experiments_DST/multiwoz2.1/t5base_lr1")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart") 
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    parser.add_argument("--is_reader_store", action='store_true', help="help judge load the input data")
    parser.add_argument("--is_ablation", action='store_true', help="help judge load the input data")
    parser.add_argument("--data_dir", type=str, default="./experiments_DST/multiwoz2.1/data_v1", help="store the data for different version input")
    parser.add_argument("--logger_file_name", type=str, default="log", help="log name for logger")
    parser.add_argument("--threhold", type=float, default=0.5, help="the threhold of judge the classification")
    parser.add_argument("--is_v20", action='store_true', help="help judge the version of input data")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.is_v20:
        old_version = "2.1"
        new_version = "2.0"
        cfg.data_path = cfg.data_path.replace(old_version, "")
        cfg.dev_list = cfg.dev_list.replace(old_version, "")
        cfg.test_list = cfg.test_list.replace(old_version, "")
        args.data_dir = args.data_dir.replace(old_version, new_version)
    print(cfg.data_path)
    print(cfg.dev_list)
    print(cfg.test_list)
    print(args.data_dir)

    cfg.batch_size = args.batch_size
    set_cfg_batch_size(args.batch_size)

    cfg.seed = args.seed
    cfg.mode = args.mode

    if args.mode == 'test':
        parse_arg_cfg(args)
        logger.info("testing ................................................")
        cfg_load = json.loads(open(os.path.join(args.model_path, 'exp_cfg.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode']:
                continue
            setattr(cfg, k, v)
            cfg.result_path = os.path.join(args.model_path, 'result.csv')
    else:
        parse_arg_cfg(args)
        if args.model_path == "":
            args.model_path = 'experiments_DST/dataV2{}_sd{}_lr{}_bs{}_sp{}_dc{}_cw{}_model_{}_noupdate{}/'.format(
                '-'.join(cfg.exp_domains), cfg.seed, args.lr, cfg.batch_size,
                cfg.early_stop_count, args.lr_decay, args.context_window, args.pretrained_checkpoint, args.noupdate_dst)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        cfg.result_path = os.path.join(args.model_path, 'result.csv')
        cfg.eval_load_path = args.model_path

    file_handler = logging.FileHandler(os.path.join(args.model_path, "%s.txt" % (args.logger_file_name)))
    logger.addHandler(file_handler)
    logger.info(args)

    logger.info("model path is {} ".format(args.model_path))
    logger.info("random seed {}, early stop count {}".format(cfg.seed, cfg.early_stop_count))
    # cfg._init_logging_handler(args.mode)
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    #cfg.model_parameters = m.count_params()

    # logging.info(str(cfg))

    if args.mode == 'train':
        with open(os.path.join(args.model_path, 'exp_cfg.json'), 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)
        m = Model(args)
        logger.info("total trainable params: {}".format(m.count_params()))
        m.train()
    elif args.mode == 'test':
        logger.info("start testing .............................................")
        m = Model(args, test=True)
        # print(m.model.get_input_embeddings())
        logger.info("total trainable params: {}".format(m.count_params()))
        m.eval(data='test')


if __name__ == '__main__':
    main()
