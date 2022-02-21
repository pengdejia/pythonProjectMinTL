# !usr/bin/env python 
# -*- coding:utf-8 _*-
"""
@Author:Derrick
 
@File:model_dst.py
@Time:2022/2/19 14:24
 
"""
import torch
import torch.nn as nn
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration,
                          WEIGHTS_NAME, CONFIG_NAME, get_linear_schedule_with_warmup)

from damd_multiwoz import ontology
print("new code version")


class BartTokenizer(BartTokenizer):
    def encode(self, text, add_special_tokens=False):
        encoded_inputs = self.encode_plus(text, add_special_tokens=False)
        return encoded_inputs["input_ids"]


class BART_DST(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def inference(
            self,
            tokenizer,
            reader,
            prev,
            input_ids=None,
            attention_mask=None,
            turn_domain=None,
    ):
        dst_outputs = self.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    eos_token_id=tokenizer.encode("<eos_b>")[0],
                                    decoder_start_token_id=self.config.decoder_start_token_id,
                                    max_length=200,
                                    min_length=1,
                                    num_beams=1,
                                    length_penalty=1.0,
                                    )
        dst_outputs = dst_outputs.tolist()
        # DST_UPDATE -> DST
        # check whether need to add eos
        # dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]
        constraint_dict_updates = [reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i])) for i in
                                   range(batch_size)]

        if prev['bspn']:
            # update the belief state
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i]) for i in
                           range(batch_size)]

        return dst_outputs


class T5_DST(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def inference(
            self,
            tokenizer,
            reader,
            prev,
            input_ids=None,
            attention_mask=None,
            turn_domain=None,
    ):
        dst_outputs = self.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    eos_token_id=tokenizer.encode("<eos_b>")[0],
                                    decoder_start_token_id=self.config.decoder_start_token_id,
                                    max_length=200,
                                    )
        dst_outputs = dst_outputs.tolist()
        # DST_UPDATE -> DST
        # check whether need to add eos
        # dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]
        # constraint_dict_updates = [reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i])) for i in range(batch_size)]

        if prev['bspn']:
            # update the belief state
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i]) for i in
                           range(batch_size)]

        return dst_outputs


class T5_DST_v2(nn.Module):
    def __init__(self, args, test=False):
        super(T5_DST_v2, self).__init__()
        if test:
            self.T5 = T5ForConditionalGeneration.from_pretrained(args.model_path)
        else:
            self.T5 = T5ForConditionalGeneration.from_pretrained(args.pretrained_checkpoint)
        if "t5-small" in args.pretrained_checkpoint:
            dim = 512
        if "t5-base" in args.pretrained_checkpoint:
            dim = 768
        self.fc = nn.Linear(dim, len(ontology.all_domain_slots))
        self.dropout = nn.Dropout(0.1)
        self.crition1 = torch.nn.BCEWithLogitsLoss()
        self.predict_possible = torch.nn.Sigmoid()
        self.threhold = args.threhold
        self.mode = args.mode

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            lm_labels=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            head_mask=None,
            return_dict=None,
            labels_tensor=None
    ):
        # batch_size seq_len hidden_dim
        encoder_outputs = self.T5.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # print(encoder_outputs[0].shape)
        if labels_tensor is not None:
            # batch size hidden_dim
            batch_sentence_tensor = torch.mean(encoder_outputs[0], dim=1)
            # batch size labels
            perdict_labels_tensor = self.fc(batch_sentence_tensor)
            labels_loss = self.crition1(perdict_labels_tensor.float(), labels_tensor.float())
        else:
            # print("yes")
            labels_loss = None

        dst_outputs = self.T5(encoder_outputs=encoder_outputs,
                              decoder_input_ids=decoder_input_ids,
                              lm_labels=lm_labels,
                              )
        dst_loss = dst_outputs[0]
        return dst_loss, labels_loss

    def inference(
            self,
            tokenizer,
            reader,
            prev,
            input_ids=None,
            attention_mask=None,
            turn_domain=None,
            labels_tensor=None,
    ):
        multi_label_true_list = list()
        multi_label_count = 0
        predict_labels_lists = None
        gold_labels_lists = None
        if self.mode == "test":
            encoder_outputs = self.T5.encoder(input_ids=input_ids, attention_mask=attention_mask)
            batch_sentence_tensor = torch.mean(encoder_outputs[0], dim=1)
            predict_labels_tensor = self.fc(batch_sentence_tensor).float()
            predict_labels_tensor = self.predict_possible(predict_labels_tensor)
            x = torch.ones_like(predict_labels_tensor, dtype=torch.long).to(predict_labels_tensor.device)
            y = torch.zeros_like(predict_labels_tensor, dtype=torch.long).to(predict_labels_tensor.device)
            predict_labels_tensor = torch.where(predict_labels_tensor > self.threhold, x, y)

            for i in range(predict_labels_tensor.shape[0]):
                if labels_tensor[i, :].equal(predict_labels_tensor[i, :]):
                    multi_label_true_list.append(1)
                    multi_label_count += 1
                else:
                    multi_label_true_list.append(0)
            predict_labels_lists = list()
            predict_labels_lists.extend(reader.get_domain_slot_list(predict_labels_tensor.cpu().numpy()))
            gold_labels_lists = list()
            gold_labels_lists.extend(reader.get_domain_slot_list(labels_tensor.cpu().numpy()))
            # for i in range(len(predict_labels_lists)):
            #     print(predict_labels_lists[i])
            #     print(gold_labels_lists[i])


        dst_outputs = self.T5.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       eos_token_id=tokenizer.encode("<eos_b>")[0],
                                       decoder_start_token_id=self.T5.config.decoder_start_token_id,
                                       max_length=200,
                                       )
        dst_outputs = dst_outputs.tolist()
        # DST_UPDATE -> DST
        # check whether need to add eos
        # dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]

        if prev['bspn']:
            # update the belief state
            # candidate list: ontology.all_domain_slots gold_labels_lists[i] predict_labels_lists[i]
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i],
                                              domain_slot_list=ontology.all_domain_slots) for i in
                           range(batch_size)]
            # print(dst_outputs)
        return dst_outputs, multi_label_count, multi_label_true_list
