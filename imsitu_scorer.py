#original code https://github.com/my89/imSitu/blob/master/imsitu.py

import torch

class imsitu_scorer():
    def __init__(self, encoder,topk, nref, image_group = {}):
        self.score_cards = {}
        self.topk = topk
        self.nref = nref
        self.image_group = image_group
        self.encoder = encoder

    def clear(self):
        self.score_cards = {}

    def add_point(self, verb_predict, gt_verbs, gt_labels, image_names = None):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            #label_pred = labels_predict[i]
            gt_label = gt_labels[i]
            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = torch.max(gt_verb, 0)[1]
            #print('groud truth verb id:', gt_v)

            if image_names is not None: _image = image_names[i]

            if image_names is not None and _image in self.image_group: sc_key = (gt_v, self.image_group[_image])
            else: sc_key = (gt_v, "")

            if sc_key not in self.score_cards:
                new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
                self.score_cards[sc_key] = new_card

            score_card = self.score_cards[sc_key]
            score_card["n_image"] += 1

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            score_card["n_value"] += gt_role_count

            '''all_found = True

            for k in range(gt_role_count):
                label_id = torch.max(label_pred[k],0)[1]
                #print('predicted label id', label_id)
                found = False
                for r in range(0,self.nref):
                    gt_label_id = torch.max(gt_label[r][k], 0)[1]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                if found and verb_found: score_card["value"] += 1
                if found: score_card["value*"] += 1

            if all_found and verb_found: score_card["value-all"] += 1
            if all_found: score_card["value-all*"] += 1'''

    def combine(self, rv, card):
        for (k,v) in card.items(): rv[k] += v

    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        agg_cards = {}
        for (key, card) in self.score_cards.items():
            (v,g) = key
            if len(groups) == 0 or g in groups:
                if v not in agg_cards:
                    new_card = {"verb":0.0, "n_image":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
                    agg_cards[v] = new_card
                self.combine(agg_cards[v], card)
        nverbs = len(agg_cards)
        for (v, card) in agg_cards.items():
            img = card["n_image"]
            nvalue = card["n_value"]
            rv["verb"] += card["verb"]/img
            rv["value-all"] += card["value-all"]/img
            rv["value-all*"] += card["value-all*"]/img
            rv["value"] += card["value"]/nvalue
            rv["value*"] += card["value*"]/nvalue

        rv["verb"] /= nverbs
        rv["value-all"] /= nverbs
        rv["value-all*"] /= nverbs
        rv["value"] /= nverbs
        rv["value*"] /= nverbs

        return rv
