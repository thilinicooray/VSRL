import torch
from imsitu_encoder_new import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_scorer_updated import imsitu_scorer
import json
from models import cnn_gcn_model1
import os
from models import utils
#from torchviz import make_dot
from graphviz import Digraph

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, eval_frequency=500):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 50
    dev_score_list = []

    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)

    '''print('init param data check :')
    for f in model.parameters():
        if f.requires_grad:
            print(f.data.size())'''


    for epoch in range(max_epoch):
        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        mx = len(train_loader)
        for i, (img, verb, roles,labels) in enumerate(train_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            total_steps += 1

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()

            verb_predict, role_predict = model(img, verb, roles)

            '''g = make_dot(verb_predict, model.state_dict())
            g.view()'''

            loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
            #print('current loss = ', loss)

            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1)


            '''for param in filter(lambda p: p.requires_grad,model.parameters()):
                print(param.grad.data.sum())'''

            #start debugger
            #import pdb; pdb.set_trace()


            optimizer.step()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.data.item()

            top1.add_point(verb_predict, verb, role_predict, labels, roles)
            top5.add_point(verb_predict, verb, role_predict, labels, roles)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.data[0],
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))
                print('Dev loss :', val_loss)

                dev_score_list.append(avg_score)
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    checkpoint_name = os.path.join(model_dir, '{}_devloss_cnngraph_{}.h5'.format('baseline', len(dev_score_list)))
                    utils.save_net(checkpoint_name, model)
                    print ('New best model saved! {0}'.format(max_score))

                #eval on the trainset

                top1, top5, val_loss = eval(model, traindev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('TRAINDEV {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))

                print('current train loss', train_loss)
                train_loss = 0
                top1 = imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer(encoder, 5, 3)

            del verb_predict, role_predict, loss, img, verb, roles, labels
            #break
        scheduler.step()
        print('Epoch ', epoch, ' completed!')
        #break

def eval(model, dev_loader, encoder, gpu_mode):
    model.eval()
    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img, verb, roles,labels) in enumerate(dev_loader):
            #print("{}/{} batches\r".format(i+1,mx)) ,
            '''im_data = torch.squeeze(im_data,0)
            im_info = torch.squeeze(im_info,0)
            gt_boxes = torch.squeeze(gt_boxes,0)
            num_boxes = torch.squeeze(num_boxes,0)
            verb = torch.squeeze(verb,0)
            roles = torch.squeeze(roles,0)
            labels = torch.squeeze(labels,0)'''

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            verb_predict, role_predict = model(img, verb, roles)
            loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.data[0]
            top1.add_point(verb_predict, verb, role_predict, labels, roles)
            top5.add_point(verb_predict, verb, role_predict, labels, roles)

            del verb_predict, role_predict, img, verb, roles, labels, loss

    return top1, top5, val_loss/mx

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)

    args = parser.parse_args()

    dataset_folder = 'imSitu'
    imgset_folder = 'resized_256'

    train_set = json.load(open(dataset_folder + "/train.json"))
    encoder = imsitu_encoder(train_set)

    model = cnn_gcn_model1.baseline(encoder, args.gpuid)

    train_set = imsitu_loader(imgset_folder, train_set, encoder, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader(imgset_folder, dev_set, encoder, model.train_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=4)

    traindev_set = json.load(open(dataset_folder +"/train_eval.json"))
    traindev_set = imsitu_loader(imgset_folder, traindev_set, encoder, model.train_preprocess())
    traindev_loader = torch.utils.data.DataLoader(traindev_set, batch_size=64, shuffle=True, num_workers=4)



    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()
    lr_set = [0.001]
    decay_set = [5e-4]

    for lr in lr_set:
        for decay in decay_set:
            #lr, weight decay user param
            print('CURRENT PARAM SET : lr, decay :' , lr, decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)
            #gradient clipping, grad check

            print('Model training started!')
            train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, 200, 'trained_models', encoder, args.gpuid)



if __name__ == "__main__":
    main()






