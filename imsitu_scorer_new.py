#original code https://github.com/my89/imSitu/blob/master/imsitu.py

import torch

class imsitu_scorer():
    def __init__(self, encoder,topk, nref):
        self.score_cards = []
        self.topk = topk
        self.nref = nref
        self.encoder = encoder

    def clear(self):
        self.score_cards = {}

    def add_point(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]
            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]


            gt_v = torch.max(gt_verb, 0)[1]
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            score_card["n_value"] += gt_role_count

            all_found = True

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
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def combine(self, rv, card):
        for (k,v) in card.items(): rv[k] += v

    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        total_len = len(self.score_cards)
        for card in self.score_cards:
            rv["verb"] += card["verb"]
            rv["value-all"] += card["value-all"]
            rv["value-all*"] += card["value-all*"]
            rv["value"] += card["value"]
            rv["value*"] += card["value*"]

        rv["verb"] /= total_len
        rv["value-all"] /= total_len
        rv["value-all*"] /= total_len
        rv["value"] /= total_len
        rv["value*"] /= total_len

        return rv
