import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_scorer import imsitu_scorer
import json
from models import cnn_graph_baseline
import os
from models import utils

def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, eval_frequency=5000):
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

            loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
            #print('current loss = ', loss)

            loss.backward()
            optimizer.step()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.data.item()

            top1.add_point(verb_predict, verb, role_predict, labels)
            top5.add_point(verb_predict, verb, role_predict, labels)


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

                print('current train loss', train_loss)
                train_loss = 0
                top1 = imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer(encoder, 5, 3)

            del verb_predict, role_predict, loss, img, verb, roles, labels
            #break
        scheduler.step()
        print('Epoch ', epoch, ' completed!')
        break

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
            top1.add_point(verb_predict, verb, role_predict, labels)
            top5.add_point(verb_predict, verb, role_predict, labels)

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

    model = cnn_graph_baseline.baseline(encoder, args.gpuid)

    train_set = imsitu_loader(imgset_folder, train_set, encoder, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader(imgset_folder, dev_set, encoder, model.train_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=4)



    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()
    lr_set = [0.001]
    decay_set = [5e-4]

    for lr in lr_set:
        for decay in decay_set:
            #lr, weight decay user param
            print('CURRENT PARAM SET : lr, decay :' , lr, decay)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
            #gradient clipping, grad check

            print('Model training started!')
            train(model, train_loader, dev_loader, optimizer, scheduler, 200, 'trained_models', encoder, args.gpuid)



if __name__ == "__main__":
    main()






