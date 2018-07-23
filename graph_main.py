import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_scorer import imsitu_scorer
import json
from models import graph_baseline
import os
from models import utils

def train(model, train_loader, dev_loader, optimizer, max_epoch, model_dir, encoder, gpu_mode, eval_frequency=500):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 10
    dev_loss_list = []

    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)

    '''print('init param data check :')
    for f in model.parameters():
        print('init data and size')
        print(f.data)
        print(f.data.size())'''


    for epoch in range(max_epoch):
        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        for i, (im_data, im_info, gt_boxes, num_boxes, verb, roles, labels) in enumerate(train_loader):
            total_steps += 1
            im_data = torch.squeeze(im_data,0)
            im_info = torch.squeeze(im_info,0)
            gt_boxes = torch.squeeze(gt_boxes,0)
            num_boxes = torch.squeeze(num_boxes,0)
            verb = torch.squeeze(verb,0)
            roles = torch.squeeze(roles,0)
            labels = torch.squeeze(labels,0)

            print('batch details \n\timdata: ', im_data.size())
            print('\tim_info: ', im_info.size())
            print('\tgt_boxes: ', gt_boxes.size())
            print('\tnum_boxes: ', num_boxes.size())
            print('\tverb: ', verb.size())
            print('\troles: ', roles.size())
            print('\tlabels: ', labels.size())

            if gpu_mode >= 0:
                im_data = torch.autograd.Variable(im_data.cuda())
                im_info = torch.autograd.Variable(im_info.cuda())
                gt_boxes = torch.autograd.Variable(gt_boxes.cuda())
                num_boxes = torch.autograd.Variable(num_boxes.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                im_data = torch.autograd.Variable(im_data)
                im_info = torch.autograd.Variable(im_info)
                gt_boxes = torch.autograd.Variable(gt_boxes)
                num_boxes = torch.autograd.Variable(num_boxes)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            verb_predict, role_predict = model(im_data, im_info, gt_boxes, num_boxes, verb, roles)

            loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
            print('current loss = ', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.data

            top1.add_point(torch.unsqueeze(verb_predict,0), torch.unsqueeze(verb,0),
                           torch.unsqueeze(role_predict,0), torch.unsqueeze(labels,0))
            top5.add_point(torch.unsqueeze(verb_predict,0), torch.unsqueeze(verb,0),
                           torch.unsqueeze(role_predict,0), torch.unsqueeze(labels,0))


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.data[0],
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, dev_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))

                dev_loss_list.append(dev_loss)
                min_loss = min(dev_loss_list)

                if min_loss == dev_loss_list[-1]:
                    checkpoint_name = os.path.join(model_dir, '{}_devloss_{}.h5'.format('baseline', len(dev_loss_list)))
                    utils.save_net(checkpoint_name, model)
                    print ('New best model saved! {0}'.format(min_loss))

                print('current train loss', train_loss)
                train_loss = 0

        print('Epoch ', epoch, ' completed!')

def eval(model, dev_loader, encoder, gpu_mode):
    model.eval()
    dev_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)
    mx = len(dev_loader)
    for i, (img, verb, roles, labels) in enumerate(dev_loader):
        print("{}/{} batches\r".format(i+1,mx)) ,
        if gpu_mode >= 0:
            img = torch.autograd.Variable(img.cuda())
            verb = torch.autograd.Variable(verb.cuda())
            roles = torch.autograd.Variable(roles.cuda())
            labels = torch.autograd.Variable(labels.cuda())
        else:
            img = torch.autograd.Variable(img)
            verb = torch.autograd.Variable(verb)
            roles = torch.autograd.Variable(roles)
            labels = torch.autograd.Variable(labels)
        #todo: implement beam search for eval mode
        verb_predict, role_predict = model(img, verb, roles)

        loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
        dev_loss += loss.data

        top1.add_point(verb_predict, verb, role_predict, labels)
        top5.add_point(verb_predict, verb, role_predict, labels)

    return top1, top5, dev_loss

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument('--cnn_pretrained', default='/home/thilini/sem-img/VSRL/pretrained_models/faster_rcnn_1_11_416.pth')

    args = parser.parse_args()

    dataset_folder = 'imSitu'
    imgset_folder = 'resized_256'

    train_set = json.load(open(dataset_folder + "/train.json"))
    encoder = imsitu_encoder(train_set)

    train_set = imsitu_loader(imgset_folder, train_set, encoder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=3)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader(imgset_folder, dev_set, encoder)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=3, shuffle=True, num_workers=3)

    model = graph_baseline.baseline(encoder, args.gpuid, args.cnn_pretrained)

    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()

    #lr, weight decay user param
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.01, weight_decay=5e-4)
    #gradient clipping, grad check

    print('Model training started!')
    train(model, train_loader, dev_loader, optimizer,200, 'trained_models', encoder, args.gpuid)



if __name__ == "__main__":
    main()






