# -*- coding: utf-8 -*-
"""Code including data preprocessing, loss computation, vector representation output, etc for document relation
module."""

import numpy as np
import torch

from dee.event_type import event_type_fields_list

CLS_ID = 101
SEP_ID = 102

HEAD_DICT = {
    'EquityHolder': 0, 'Pledger': 1, 'PledgedShares': 2, 'FrozeShares': 3, 'TotalHoldingShares': 4,
    'Pledgee': 5, 'StartDate': 6, 'LegalInstitution': 7, 'TotalHoldingRatio': 8, 'CompanyName': 9,
    'TradedShares': 10, 'EndDate': 11, 'HighestTradingPrice': 12, 'TotalPledgedShares': 13, 'LowestTradingPrice': 14,
    'RepurchasedShares': 15, 'LaterHoldingShares': 16, 'ClosingDate': 17
}

TAIL_DICT = {
    'EndDate': 0, 'StartDate': 1, 'ReleasedDate': 2, 'UnfrozeDate': 3, 'TotalHoldingRatio': 4, 'TotalHoldingShares': 5,
    'AveragePrice': 6, 'TotalPledgedShares': 7, 'RepurchaseAmount': 8, 'LaterHoldingShares': 9, 'ClosingDate': 10,
    'RepurchasedShares': 11, 'LegalInstitution': 12, 'Pledgee': 13, 'LowestTradingPrice': 14, 'TradedShares': 15,
    'FrozeShares': 16, 'PledgedShares': 17, 'HighestTradingPrice': 18
}


class DocREDInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, ent_mask, ent_ner, ent_pos, ent_distance, structure_mask, label=None, label_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent_mask = ent_mask
        self.ent_ner = ent_ner
        self.ent_pos = ent_pos
        self.ent_distance = ent_distance
        self.structure_mask = structure_mask
        self.label = label
        self.label_mask = label_mask


class DREProcessor:
    def __init__(self, doc_max_length=512, max_ent_cnt=42, label_map=None, token_ner_label_list=None):
        self.doc_max_length = doc_max_length
        self.max_ent_cnt = max_ent_cnt
        self.label_map = label_map
        self.idx_label_map = {v: k for k, v in self.label_map.items()}
        self.tok_label_ent_label_map, self.ent_idx_label_map = generate_tok_label_to_label_map(token_ner_label_list)

    @staticmethod
    def prepare_input_tensor_for_re(features, device):
        return {"input_ids": torch.tensor([feature.input_ids for feature in features], device=device, dtype=torch.long),
                "attention_mask": torch.tensor([feature.attention_mask for feature in features], device=device, dtype=torch.long),
                "token_type_ids": torch.tensor([feature.token_type_ids for feature in features], device=device, dtype=torch.long),
                "ent_mask": torch.tensor([feature.ent_mask for feature in features], device=device, dtype=torch.float),
                "ent_ner": torch.tensor([feature.ent_ner for feature in features], device=device, dtype=torch.long),
                "ent_pos": torch.tensor([feature.ent_pos for feature in features], device=device, dtype=torch.long),
                "ent_distance": torch.tensor([feature.ent_distance for feature in features], device=device, dtype=torch.long),
                "structure_mask": torch.tensor([feature.structure_mask for feature in features], device=device, dtype=torch.bool),
                "label": torch.tensor([feature.label for feature in features], device=device, dtype=torch.long)
                if features[0].label is not None else None,
                "label_mask": torch.tensor([feature.label_mask for feature in features], device=device, dtype=torch.bool)
                }

    def convert_ner_result_to_dre_feature(self, doc_fea_list, doc_span_info_list, use_bert=True, label_map=None):
        """Convert output from NER module into the input of dre module."""
        features = []
        for doc_span_info, doc_fea in zip(doc_span_info_list, doc_fea_list):
            doc_token_ids_list = doc_fea.doc_token_ids
            event_dag_info = doc_span_info.event_dag_info
            # Generate distance buckets
            distance_buckets = np.zeros(self.doc_max_length, dtype='int64')
            distance_buckets[1] = 1
            distance_buckets[2:] = 2
            distance_buckets[4:] = 3
            distance_buckets[8:] = 4
            distance_buckets[16:] = 5
            distance_buckets[32:] = 6
            distance_buckets[64:] = 7
            distance_buckets[128:] = 8
            distance_buckets[256:] = 9

            # prepare input_ids, concatenate sentence-level input_ids to doc-level input_ids.
            doc_token_ids_list = doc_token_ids_list.tolist()
            input_ids = []
            tok_to_sent = []
            tok_to_word = []
            for sent_id, sent_input_ids in enumerate(doc_token_ids_list):
                # if ner_model use bert, then consider cls and sep token, otherwise, not consider them.
                if use_bert:
                    cls_pos = sent_input_ids.index(CLS_ID)
                    sep_pos = sent_input_ids.index(SEP_ID)
                    valid_sent_input_ids = sent_input_ids[cls_pos + 1: sep_pos]
                else:
                    valid_sent_input_ids = sent_input_ids
                input_ids += valid_sent_input_ids
                tok_to_sent += [sent_id] * len(valid_sent_input_ids)
                tok_to_word += list(range(len(valid_sent_input_ids)))

            if len(input_ids) < self.doc_max_length - 2:
                input_ids = [CLS_ID] + input_ids + [SEP_ID]
                tok_to_sent = [None] + tok_to_sent + [None]
                tok_to_word = [None] + tok_to_word + [None]
                attention_mask = [1] * len(input_ids)
                # padding
                none_padding = [None] * (self.doc_max_length - len(input_ids))
                tok_to_word += none_padding
                tok_to_sent += none_padding
                zero_padding = [0] * (self.doc_max_length - len(input_ids))
                input_ids += zero_padding
                attention_mask += zero_padding

            else:
                input_ids = [CLS_ID] + input_ids[:self.doc_max_length - 2] + [SEP_ID]
                tok_to_sent = [None] + tok_to_sent[:self.doc_max_length - 2] + [None]
                tok_to_word = [None] + tok_to_word[:self.doc_max_length - 2] + [None]
                attention_mask = [1] * len(input_ids)

            token_type_ids = [0] * self.doc_max_length

            # ent_mask & ner / coreference feature.
            vertex_set = self.generate_vertex_set(doc_span_info.mention_drange_list,
                                                  doc_span_info.span_mention_range_list,
                                                  doc_span_info.mention_type_list)
            ent_mask = np.zeros((self.max_ent_cnt, self.doc_max_length), dtype='int64')
            ent_ner = [0] * self.doc_max_length
            ent_pos = [0] * self.doc_max_length
            tok_to_ent = [-1] * self.doc_max_length

            for ent_idx, ent in enumerate(vertex_set):
                for mention in ent:
                    # mention has following info: [sent_id, head_id, tail_id, entity_type]
                    for tok_idx in range(len(input_ids)):
                        if tok_to_sent[tok_idx] == mention[0] \
                                and mention[1] <= tok_to_word[tok_idx] < mention[2]:
                            ent_mask[ent_idx][tok_idx] = 1
                            ent_ner[tok_idx] = mention[3]
                            ent_pos[tok_idx] = ent_idx + 1
                            tok_to_ent[tok_idx] = ent_idx

            # entity relative distance feature
            ent_first_appearance = [0] * self.max_ent_cnt
            ent_distance = np.zeros((self.max_ent_cnt, self.max_ent_cnt), dtype='int8')  # padding id is 10
            for i in range(len(vertex_set)):
                if np.all(ent_mask[i] == 0):
                    continue
                else:
                    ent_first_appearance[i] = np.where(ent_mask[i] == 1)[0][0]
            for i in range(len(vertex_set)):
                for j in range(len(vertex_set)):
                    if ent_first_appearance[i] != 0 and ent_first_appearance[j] != 0:
                        if ent_first_appearance[i] >= ent_first_appearance[j]:
                            ent_distance[i][j] = distance_buckets[ent_first_appearance[i] - ent_first_appearance[j]]
                        else:
                            ent_distance[i][j] = - distance_buckets[- ent_first_appearance[i] + ent_first_appearance[j]]
            ent_distance += 10  # norm from [-9, 9] to [1, 19]

            # Generate structure attentive mask feature
            structure_mask = np.zeros((5, self.doc_max_length, self.doc_max_length), dtype='float')
            for i in range(self.doc_max_length):
                if attention_mask[i] == 0:
                    break
                else:
                    if tok_to_ent[i] != -1:
                        for j in range(self.doc_max_length):
                            if tok_to_sent[j] is None:
                                continue
                            # intra
                            if tok_to_sent[j] == tok_to_sent[i]:
                                # intra-coref
                                if tok_to_ent[j] == tok_to_ent[i]:
                                    structure_mask[0][i][j] = 1
                                # intra-relate
                                elif tok_to_ent[j] != -1:
                                    structure_mask[1][i][j] = 1
                                # intra-NA
                                else:
                                    structure_mask[2][i][j] = 1
                            # inter
                            else:
                                # inter-coref
                                if tok_to_ent[j] == tok_to_ent[i]:
                                    structure_mask[3][i][j] = 1
                                # inter-relate
                                elif tok_to_ent[j] != -1:
                                    structure_mask[4][i][j] = 1

            # label
            if event_dag_info is not None:
                labels = generate_label_info(event_dag_info, self.max_ent_cnt, label_map=label_map)
            else:
                labels = None

            label_ids = np.zeros((self.max_ent_cnt, self.max_ent_cnt), dtype=int)
            if labels is not None:
                for label in labels:
                    label_ids[label[0]][label[1]] = self.label_map[label[2]]

            # label_ids = np.zeros((self.max_ent_cnt, self.max_ent_cnt, len(self.label_map.keys())), dtype='bool')
            # # test file does not have "labels"
            # if labels is not None:
            #     for label in labels:
            #         label_ids[label[0]][label[1]][self.label_map[label[2]]] = 1
            # for h in range(len(vertex_set)):
            #     for t in range(len(vertex_set)):
            #         if np.all(label_ids[h][t] == 0):
            #             label_ids[h][t][0] = 1

            label_mask = np.zeros((self.max_ent_cnt, self.max_ent_cnt), dtype='bool')
            label_mask[:len(vertex_set), :len(vertex_set)] = 1
            # The diagonal pair is not considered.
            for ent in range(len(vertex_set)):
                label_mask[ent][ent] = 0
            for ent in range(len(vertex_set)):
                if np.all(ent_mask[ent] == 0):
                    label_mask[ent, :] = 0
                    label_mask[:, ent] = 0

            ent_mask = norm_mask(ent_mask)

            features.append(DocREDInputFeatures(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        ent_mask=ent_mask,
                        ent_ner=ent_ner,
                        ent_pos=ent_pos,
                        ent_distance=ent_distance,
                        structure_mask=structure_mask,
                        label=label_ids,
                        label_mask=label_mask,
                    ))
        return features

    @staticmethod
    def generate_structural_attentive_feature(span_mention_range_list, mention_drange_list, label_list, num_sent,
                                              device, raat_id=1, head_center=True, num_relation=19):
        """Generate structural attentive mask feature for span_mention, Three types of relation are considered:
            relate: two mention have relation;
            co-reference: two mention denotes the same span;
            NA: two mention don't have relation.
        """
        # first 18 matrix for relation-pair with different head entity; 19th for co-ref, 20th for IntraNE,
        # 21st for NA.
        relation_num = num_relation + 3
        if not span_mention_range_list:
            print("no valid span is extracted!")
            return None
        span_mention_num = span_mention_range_list[-1][1]

        # Build mention2span map
        mention2span_map = {}
        for id, span_range in enumerate(span_mention_range_list):
            for i in range(span_range[0], span_range[1]):
                mention2span_map[i] = id

        # Build relation matrix, 0 denotes two span don't have relation, 1 denotes they have relation.
        shape_tup = (num_relation, len(span_mention_range_list), len(span_mention_range_list))
        relation_mat = generate_relation_matrix(label_list, shape_tup, head_center, num_relation)

        # Build entity_sent_map
        span_sent_map = {}
        # raat_id = 1 denotes transformer-2; while raat_id = 2 denotes transformer-3
        for i, mention_drange in enumerate(mention_drange_list):
            if raat_id == 1:
                span_sent_map[i] = [mention_drange[0]]
            else:
                # after transformer-2, all same mentions pooling into one span, so there is one-to-many map
                # from span to sent.
                span_sent_map[i] = mention_drange

        # generate structural attentive mask feature
        structure_mask = np.zeros((relation_num, span_mention_num + num_sent, span_mention_num + num_sent),
                                  dtype='float')
        # structure_mask = torch.zeros((relation_num, span_mention_num + num_sent, span_mention_num + num_sent),
        #                               dtype=torch.bool, device=device)
        for i in range(span_mention_num + num_sent):
            # Consider StartDate&EndDate may share same span. id of StateDate in relation_mat is 2.
            # if i < span_mention_num and relation_mat[2, mention2span_map[i], mention2span_map[i]] == 1:
            #     structure_mask[2][i][i] = 1
            # Consider different event roles may share same span (such as StartDate&EndDate).
            if i < span_mention_num and any(relation_mat[:, mention2span_map[i], mention2span_map[i]]):
                pos = np.where(relation_mat[:, mention2span_map[i], mention2span_map[i]] == 1)[0][0]
                structure_mask[pos][i][i] = 1

            for j in range(i):
                if i >= span_mention_num and j >= span_mention_num:
                    continue
                # consider intraNE (consider info flow between the span mention and corresponding sentence)
                if i >= span_mention_num > j:
                    if j in span_sent_map and i - span_mention_num in span_sent_map[j]:
                        structure_mask[-2][i][j] = 1
                        structure_mask[-2][j][i] = 1
                # consider co-ref
                elif mention2span_map[i] == mention2span_map[j]:
                    structure_mask[-3][i][j] = 1
                    structure_mask[-3][j][i] = 1
                    span_i = mention2span_map[i]
                    # consider StartDate&EndDate may share same span. id of StateDate in relation_mat is 2.
                    # if relation_mat[2, span_i, span_i] == 1:
                    #     structure_mask[2][i][j] = 1
                    #     structure_mask[2][j][i] = 1
                    # consider different event roles may share same span (such as StartDate&EndDate)
                    if any(relation_mat[:, span_i, span_i]):
                        pos = np.where(relation_mat[:, span_i, span_i] == 1)[0][0]
                        structure_mask[pos][i][j] = 1
                        structure_mask[pos][j][i] = 1
                # consider relate (symmetry)
                else:
                    # differentiate relation information.
                    span_i = mention2span_map[i]
                    span_j = mention2span_map[j]
                    if any(relation_mat[:, span_i, span_j]):
                        pos = np.where(relation_mat[:, span_i, span_j] == 1)[0][0]
                        structure_mask[pos][i][j] = 1
                        structure_mask[pos][j][i] = 1
                    # NA
                    else:
                        structure_mask[-1][i][j] = 1
                        structure_mask[-1][j][i] = 1

        structure_mask = torch.tensor(structure_mask, device=device, dtype=torch.bool)
        return structure_mask

    @staticmethod
    def expand_structure_mask(structure_mask, expand_span_id, label_list, span_mention_num, span_sent_id_list=None,
                              head_center=True, num_relation=19):
        """Expand existing structure_mask in EDAG step. the memory path grows each step, and the
        corresponding structure_mask should grow in the same way."""
        def change_structure_mask_val(x, y, x_span_id, y_span_id, structure_mask_tensor):
            # co-ref
            if x_span_id == y_span_id:
                structure_mask_tensor[-3][x][y] = True
                structure_mask_tensor[-3][y][x] = True
                # consider StartDate&EndDate may share same span. id of StateDate in relation_mat is 2.
                # if relation_mat[2][x_span_id][y_span_id] == 1:
                #     structure_mask_tensor[2][x][y] = True
                #     structure_mask_tensor[2][y][x] = True
                # consider different entities may share same span (such as StartDate&EndDate)
                if any(relation_mat[:, x_span_id, y_span_id]):
                    pos = np.where(relation_mat[:, x_span_id, y_span_id] == 1)[0][0]
                    structure_mask_tensor[pos][x][y] = True
                    structure_mask_tensor[pos][y][x] = True
            # related
            elif any(relation_mat[:, x_span_id, y_span_id]):
                pos = np.where(relation_mat[:, x_span_id, y_span_id] == 1)[0][0]
                structure_mask_tensor[pos][x][y] = True
                structure_mask_tensor[pos][y][x] = True
            # NA
            else:
                structure_mask_tensor[-1][x][y] = True
                structure_mask_tensor[-1][y][x] = True
        # if no entity vector is added on memory path, return the origin structure_mask.
        if not expand_span_id:
            return structure_mask
        old_mask_len = structure_mask.size()[-1]
        new_mask_size = (structure_mask.size()[0],) + (old_mask_len + len(expand_span_id),
                                                       old_mask_len + len(expand_span_id))
        # new_structure_mask = torch.zeros(new_mask_size, dtype=torch.bool).to(structure_mask)
        # new_structure_mask[:, :old_mask_len, :old_mask_len] = structure_mask

        new_structure_mask = np.zeros(new_mask_size)
        new_structure_mask[:, :old_mask_len, :old_mask_len] = structure_mask.detach().cpu().numpy()

        # Consider if span_id denotes 'None' vector added in EDAG part
        if all(i is None for i in expand_span_id):
            new_structure_mask = torch.tensor(new_structure_mask, dtype=torch.bool, device=structure_mask.device)
            return new_structure_mask

        shape_tup = (num_relation, span_mention_num, span_mention_num)
        relation_mat = generate_relation_matrix(label_list, shape_tup, head_center, num_relation)

        # expand structure mask
        for i in range(old_mask_len + len(expand_span_id)):
            for j in range(i):
                # keep old part of structure_mask unchanged.
                if i < old_mask_len and j < old_mask_len or i == j:
                    continue
                elif i >= old_mask_len and j < span_mention_num:
                    i_span_id = expand_span_id[i - old_mask_len]
                    if i_span_id:
                        change_structure_mask_val(i, j, i_span_id, j, new_structure_mask)

                elif i >= old_mask_len and j >= old_mask_len:
                    i_span_id = expand_span_id[i - old_mask_len]
                    j_span_id = expand_span_id[j - old_mask_len]
                    if i_span_id and j_span_id:
                        change_structure_mask_val(i, j, i_span_id, j_span_id, new_structure_mask)

                elif i >= old_mask_len > j:
                    i_span_id = expand_span_id[i - old_mask_len]
                    if span_sent_id_list and i_span_id and j - span_mention_num in span_sent_id_list[i_span_id]:
                        new_structure_mask[-2][i][j] = 1
                        new_structure_mask[-2][j][i] = 1

        new_structure_mask = torch.tensor(new_structure_mask, dtype=torch.bool, device=structure_mask.device)
        return new_structure_mask

    def generate_vertex_set(self, mention_drange_list, span_mention_range_list, mention_type_list):
        """Get standard format of name entities of position and type info."""
        vertex_set = []
        if len(span_mention_range_list) > self.max_ent_cnt:
            span_mention_range_list = span_mention_range_list[:self.max_ent_cnt]
        for span_mention_range in span_mention_range_list:
            curr_vertex = []
            s_id, e_id = span_mention_range
            curr_mention_drange_list = mention_drange_list[s_id: e_id]
            curr_mention_type_list = mention_type_list[s_id: e_id]
            for drange, t in zip(curr_mention_drange_list, curr_mention_type_list):
                # vertex_info: [sent_id, head_pos, tail_pos, entity_type]
                # since head_pos/tail_pos in drange consider [CLS] at the beginning, consider minus 1.
                curr_vertex.append([drange[0], drange[1] - 1, drange[2] - 1] + [self.tok_label_ent_label_map[t]])
            vertex_set.append(curr_vertex)

        return vertex_set

    def output_relation_pair_candidates(self, label_masks, input_logits, doc_span_info_list):
        """decode all relation extraction pair based on model output logits."""

        preds = input_logits.detach().cpu().numpy()
        label_masks = label_masks.detach().cpu().numpy()
        re_pair_list = []
        batch_size = preds.shape[0]
        for i in range(batch_size):
            # use doc_span_info_list to get label info.
            doc_span_info = doc_span_info_list[i]
            ent_label_list = []
            curr_re_pair_list = []
            for span_mention_range in doc_span_info.span_mention_range_list:
                label_id = doc_span_info.mention_type_list[span_mention_range[0]]
                ent_label_list.append(self.ent_idx_label_map[self.tok_label_ent_label_map[label_id]])
            pred = preds[i]
            label_mask = label_masks[i]
            for h in range(self.max_ent_cnt):
                for t in range(self.max_ent_cnt):
                    if not label_mask[h][t]:
                        continue
                    label_idx = np.argmax(pred[h][t])
                    if label_idx != 0:
                        h_label = ent_label_list[h]
                        t_label = ent_label_list[t]
                        label = self.idx_label_map[label_idx]
                        re_label_h, re_label_t = label.split("2")
                        # make sure entity type and entity-relation type is compatible.
                        # if (re_label_h == h_label or
                        #     (re_label_h in ["CriminalPosition", "VictimPosition"] and h_label == "Position"))\
                        #         and (re_label_t == t_label or
                        #              (re_label_t in ["CriminalPosition", "VictimPosition"] and t_label == "Position")):
                        #     curr_re_pair_list.append([[h, h_label], [t, t_label], label])
                        # Consider some special cases in ChiFinData, e.g. different field share same span
                        # if (re_label_h == h_label or re_label_h[-4:] == h_label[-4:] == "Date" or
                        #    (re_label_h in ["EquityHolder", "Pledger"] and h_label in ["EquityHolder", "Pledger"]))\
                        #     and (re_label_t == t_label or re_label_t[-4:] == t_label[-4:] == "Date" or
                        #          (re_label_t in ["EquityHolder", "Pledger"] and t_label in ["EquityHolder", "Pledger"])):
                        #     curr_re_pair_list.append([[h, h_label], [t, t_label], label])

                        # no restriction
                        curr_re_pair_list.append([[h, h_label], [t, t_label], label])
            re_pair_list.append(curr_re_pair_list)

        return re_pair_list


def generate_relation_matrix(label_list, shape_tup, head_center=True, num_relation=19):
    relation_mat = np.zeros(shape_tup)
    for label in label_list:
        # consider symmetry
        h_id, t_id = label[0], label[1]
        h_entity, t_entity = label[2].split("2")
        if isinstance(h_id, list):
            h_id = h_id[0]
        if isinstance(t_id, list):
            t_id = t_id[0]
        if head_center:
            if HEAD_DICT[h_entity] < num_relation:
                relation_mat[HEAD_DICT[h_entity]][h_id][t_id] = 1
                relation_mat[HEAD_DICT[h_entity]][t_id][h_id] = 1
        else:
            if TAIL_DICT[t_entity] < num_relation:
                relation_mat[TAIL_DICT[t_entity]][h_id][t_id] = 1
                relation_mat[TAIL_DICT[t_entity]][t_id][h_id] = 1
    return relation_mat


def generate_label_info(event_dag_info, max_ent_cnt, label_map=None):
    labels_info_list = []
    for event_id, event_dag in enumerate(event_dag_info):
        if event_dag is not None:
            event_field = event_type_fields_list[event_id][1]
            # use last element of event_dag to generate complete event dag.
            final_event_dag = event_dag[-1]
            for k, val in final_event_dag.items():
                for v in val:
                    complete_events = list(k) + [v]
                    for h_idx in range(len(event_dag) - 1):
                        for t_idx in range(h_idx + 1, len(event_dag)):
                            ent_h_idx = complete_events[h_idx]
                            ent_t_idx = complete_events[t_idx]
                            if ent_h_idx is not None and ent_t_idx is not None:
                                if ent_t_idx >= max_ent_cnt or ent_h_idx >= max_ent_cnt:
                                    continue
                                label = f"{event_field[h_idx]}2{event_field[t_idx]}"
                                if label_map is not None and label not in label_map.keys():
                                    continue
                                if [ent_h_idx, ent_t_idx, label] not in labels_info_list:
                                    labels_info_list.append([ent_h_idx, ent_t_idx, label])

    return labels_info_list


def generate_span_sent_id_list(mention_drange_list, span_mention_range_list):
    span_sent_id_list = []
    for mention_range in span_mention_range_list:
        sent_ids = [mention_drange_list[i][0] for i in range(*mention_range)]
        span_sent_id_list.append(sent_ids)
    return span_sent_id_list


def norm_mask(input_mask):
    output_mask = np.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not np.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask


def generate_tok_label_to_label_map(tok_label_list):
    if tok_label_list is None:
        raise ValueError("token label map must be provided.")
    # since ner format is BIOES, the divisor is 4

    # deal with entity label beyond ner label (like victim)
    tgt_label_map = {0: 0}
    ent_idx_label_map = {0: "PAD"}

    divisor = 4
    for i, label in enumerate(tok_label_list):
        if i % divisor == 1:
            tgt_label_map[i] = i // divisor + 1
            label_name = label.split("-")[-1]
            ent_idx_label_map[i // divisor + 1] = label_name
        # consider single tok entity.
        elif i % divisor == 0 and i != 0:
            tgt_label_map[i] = i // divisor

    return tgt_label_map, ent_idx_label_map
