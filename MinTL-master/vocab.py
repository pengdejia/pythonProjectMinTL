# !usr/bin/env python 
# -*- coding:utf-8 _*-
"""
@Author:Derrick
 
@File:vocab.py
@Time:2022/2/19 14:33
 
"""

class Vocab(object):
    def __init__(self, model, tokenizer):
        self.special_tokens = ["pricerange", "<pad>", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","<None>"]
        #<eos_u> /<eos_r>/ <eos_b> represents the end of user/response/ bspn
        #TODO :add <eos_s> for sign the end of scheme infortionmation
        # self.attr_special_tokens = {'pad_token': '<pad>',
        #                  'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
        #             "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
        #             "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
        #             "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
        #             "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
        #             "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]","[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","<None>"]}
        self.attr_special_tokens = {'pad_token': '<pad>',
                                    'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>",
                                                                  "<eos_u>", "<eos_r>", "<eos_b>", "<eos_s>","<eos_a>", "<go_d>",
                                                                  "[restaurant]", "[hotel]", "[attraction]", "[train]",
                                                                  "[taxi]", "[police]", "[hospital]", "[general]",
                                                                  "[inform]", "[request]",
                                                                  "[nooffer]", "[recommend]", "[select]", "[offerbook]",
                                                                  "[offerbooked]", "[nobook]", "[bye]", "[greet]",
                                                                  "[reqmore]", "[welcome]",
                                                                  "[value_name]", "[value_choice]", "[value_area]",
                                                                  "[value_price]", "[value_type]", "[value_reference]",
                                                                  "[value_phone]", "[value_address]",
                                                                  "[value_food]", "[value_leave]", "[value_postcode]",
                                                                  "[value_id]", "[value_arrive]", "[value_stars]",
                                                                  "[value_day]", "[value_destination]",
                                                                  "[value_car]", "[value_departure]", "[value_time]",
                                                                  "[value_people]", "[value_stay]",
                                                                  "[value_pricerange]", "[value_department]",
                                                                  "[db_state0]", "[db_state1]", "[db_state2]",
                                                                  "[db_state3]", "[db_state4]", "<None>"]}

        self.tokenizer = tokenizer
        # logger.info("before add special tokens vocab size is ", len(self.tokenizer))
        self.vocab_size = self.add_special_tokens_(model, tokenizer)
        # logger.info("after add special tokens vocab size is ", self.vocab_size)

    def add_special_tokens_(self, model, tokenizer):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        #orig_num_tokens = model.config.vocab_size
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(self.attr_special_tokens) # doesn't add if they are already there
        if num_added_tokens > 0:
            model.T5.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
        return orig_num_tokens + num_added_tokens

    def encode(self, word):
        """ customize for damd script """
        return self.tokenizer.encode(word)[0]

    def sentence_encode(self, word_list):
        """ customize for damd script """
        return self.tokenizer.encode(" ".join(word_list))

    def decode(self, idx):
        """ customize for damd script """
        return self.tokenizer.decode(idx)

    def sentence_decode(self, index_list, eos=None):
        """ customize for damd script """
        l = self.tokenizer.decode(index_list)
        l = l.split()
        if not eos or eos not in l:
            text = ' '.join(l)
        else:
            idx = l.index(eos)
            text = ' '.join(l[:idx])
        return text