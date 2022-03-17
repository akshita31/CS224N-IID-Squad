import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, BiDAFWithChar, QANet
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from collections import Counter
from itertools import groupby
from qanetnew import QANetNew1
import random

def getModel(word_vectors,
            char_vectors,
            char_embed_dim,
            n_encoder_blocks,
            n_head,
            num_encoder_conv, 
            num_model_conv,
            output_type,
            use_bidaf_att,
            use_old_model,
            log):
    log.info('Building model...')

    if use_old_model:
        model = BiDAFWithChar(word_vectors, char_vectors, args.hidden_size)
        # model = QANet(word_mat=word_vectors, 
        # char_mat=char_vectors, 
        # n_encoder_blocks=n_encoder_blocks,
        # n_head = n_head)
    else:
        print('Num is', num_encoder_conv)
        print('Num is', num_model_conv)
        model = QANetNew1(word_mat=word_vectors,
                      char_mat=char_vectors,
                      char_embed_dim = char_embed_dim,
                      n_encoder_blocks=n_encoder_blocks,
                      n_head=n_head,
                      num_encoder_conv = num_encoder_conv,
                      num_model_conv = num_model_conv,
                      output_type=output_type,
                      use_bidaf_att=use_bidaf_att)
    return model

def weighted_avg(log_p1_models, log_p2_models, weights, args):
    print('using weighted avg ensemble')

    n_models = log_p1_models.shape[0]

    w = weights.view(1, len(weights))
    p1, p2 = log_p1_models.exp(), log_p2_models.exp()

    p1_avg = 0
    p2_avg = 0
    for i in range(n_models):
        p1_avg = p1_avg + (weights[i] * p1[i])
        p2_avg = p2_avg + (weights[i] * p2[i])

    p1 = p1_avg / torch.sum(w)
    p2 = p2_avg / torch.sum(w)

    p1 = p1/(torch.sum(p1, dim=1).view(-1,1))
    p2 = p2/(torch.sum(p2, dim=1).view(-1,1))

    starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
    return starts, ends

def majority_voting(log_p1_models, log_p2_models, weights, args):
    #print('using majority voting ensemble')

    n_models = log_p1_models.shape[0]
    batch_size = log_p1_models.shape[1]

    w = weights.view(1, len(weights))
    p1, p2 = log_p1_models.exp(), log_p2_models.exp()

    preds = []  # (batch, n_models)
    for i in range(batch_size):
        starts, ends = util.discretize(p1[:,i], p2[:,i], args.max_ans_len, args.use_squad_v2)
        # print(starts.shape, ends.shape) # (n_models, )
        starts = starts.tolist()
        ends = ends.tolist()

        tuples = [(starts[i], ends[i]) for i in range(len(starts))] # (n_models) tuples

        preds.append(tuples)
    
    # print(preds)

    ans_starts = []
    ans_ends = []
    for i in range(batch_size):
        preds_i = preds[i] # (n_models, 2)
        # print(preds_i)        
        sorted_ct_tuples = Counter(preds_i).most_common()
        
        # ans_starts.append(sorted_ct_tuples[0][0][0])
        # ans_ends.append(sorted_ct_tuples[0][0][1])

        max_freq = sorted_ct_tuples[0][1]
        ans_choices = [span for span,ct in sorted_ct_tuples if ct == max_freq]
        ans = random.choice(ans_choices)
        ans_starts.append(ans[0])
        ans_ends.append(ans[1])
        
        


    # print("answers computed")
    # print(ans_starts, ans_ends)
    return torch.tensor(ans_starts), torch.tensor(ans_ends) # (batch, 2)


def ensemble(log_p1_models, log_p2_models, f1_scores, ensemble_method, args):
    # Perform ensemble and select starts and end indexes for whole batch, combinging probs from each model.
    # Discretize will be called in this method.
    # shape log_p1_models : (n_models, batch_size, seq_len)
    ans_starts = []
    ans_edns = []
    # print(log_p1_models.shape, log_p2_models.shape)
    n_models = len(log_p1_models)
    batch_size = len(log_p1_models[0])

    f1_scores = torch.tensor(f1_scores)



    # for i in range(batch_size):
    #     # for ith data point, get probs for each model.
    #     log_p1_model = log_p1_models[:, i]
    #     log_p2_model = log_p2_models[:, i]

    #     p1, p2 = log_p1_model.exp(), log_p2_model.exp()

    #     starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

    if ensemble_method == 'weighted_avg':
        return weighted_avg(log_p1_models, log_p2_models, weights=f1_scores, args=args)
    if ensemble_method == "majority_voting":
        return majority_voting(log_p1_models, log_p2_models, weights=f1_scores, args=args)


    # select 1st model for now.
    log_p1 = log_p1_models[0]
    log_p2 = log_p2_models[0]
    p1, p2 = log_p1.exp(), log_p2.exp()
    starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

    return starts, ends


def main(args_list, f1_scores, ensemble_method='weighted_avg'):
    
    # common args, pull from first configuration.
    args = args_list[0]

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)
    models = []

    for args_model in args_list:
        # Get model
        print(args_model)
        word_vectors2 = torch.rand((args_model.n_words, 300))
        model = getModel(word_vectors2, 
                    char_vectors,
                    char_embed_dim = args_model.char_embed_dim,
                    n_encoder_blocks= args_model.n_encoder_blocks,
                    n_head= args_model.n_head,
                    num_encoder_conv=args_model.num_encoder_conv,
                    num_model_conv=args_model.num_model_conv,
                    output_type= args_model.output_type,
                    use_bidaf_att=args_model.use_bidaf_att,
                    use_old_model = args_model.use_old_model,
                    log=log)

        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args_model.load_path}...')
        model = util.load_model(model, args_model.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            log_p1_models = torch.tensor([]).to(device)
            log_p2_models = torch.tensor([]).to(device)
            loss_models = []

            for model in models:
                # Forward
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                y1, y2 = y1.to(device), y2.to(device)

                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            
                # Add the log probs and losses from each model to these lists.
                log_p1_models = torch.cat((log_p1_models, log_p1.unsqueeze(0)), dim=0)
                log_p2_models = torch.cat((log_p2_models, log_p2.unsqueeze(0)), dim=0)
                loss_models.append(loss)

            starts, ends =  ensemble(log_p1_models, log_p2_models, f1_scores=f1_scores, ensemble_method=ensemble_method, args=args)

            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            # p1, p2 = log_p1.exp(), log_p2.exp()
            # starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)
            
            # print(starts, ends)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    checkpoints = ["/home/azureuser/CS224N-IID-Squad/save/train/qanet-ConditionalAttention-CharEmbed-200/best.pth.tar",
                    "/home/azureuser/CS224N-IID-Squad/save/train/qanetnew-light/best.pth.tar",
                    "/home/azureuser/CS224N-IID-Squad/save/train/qanet-encoder-blocks-5/best.pth.tar",
                    "/home/azureuser/CS224N-IID-Squad/save/train/qanetnew-Encoder-5-Head-4-CharEmbed200-01/best.pth.tar",
                    "/home/azureuser/CS224N-IID-Squad/save/train/qanetnew-Encoder-7-Head-8-CharEmbed200-01/best.pth.tar",
                    "/home/azureuser/CS224N-IID-Squad/save/train/CharEmbedding200d-WithBatchNorm-Relu-02/best.pth.tar"]

    f1_scores=[69.02, 68.25, 67.95, 68.22, 69.76, 66.88 ]

    num_models = len(checkpoints)
    args_list = []
    for i in range(num_models):
        args = get_test_args()
        args.load_path = checkpoints[i]

        # Override some args for each model/
        if i == 0:
            args.n_encoder_blocks = 5
            args.n_head = 8
            args.char_embed_dim = 200
            args.output_type = 'conditional_attention'
        elif i == 1:
            args.n_encoder_blocks = 5
            args.n_head = 4
            args.char_embed_dim = 128
            args.output_type = 'conditional_attention'
            args.num_encoder_conv = 2
            args.num_model_conv = 1
        elif i == 2:
            args.n_encoder_blocks = 5
            args.n_head = 8
            args.char_embed_dim = 128
            args.output_type = 'default'
        elif i == 3:
            args.n_encoder_blocks = 5
            args.n_head = 4
            args.char_embed_dim = 200
            args.output_type = 'conditional_attention'
        elif i == 4:
            args.n_encoder_blocks = 7
            args.n_head = 8
            args.char_embed_dim = 200
            args.output_type = 'conditional_attention'
        elif i == 5:
            args.use_old_model = True
            
            #args.n_words = 88714
        #     args.use_char_emb = True
        #     args.use_attention = True
        #     args.use_dynamic_decoder = False
        # elif i == 2:
        #     args.use_char_emb = False
        #     args.use_attention = False
        #     args.use_dynamic_decoder = False
        # elif i == 3:
        #     args.use_char_emb = True
        #     args.use_attention = True
        #     args.use_dynamic_decoder = False
        #     args.use_2_conv_filters = False
        # elif i == 4:
        #     args.use_char_emb = True
        #     args.use_attention = True
        #     args.use_dynamic_decoder = False
        #     args.use_multihead = True
            
        args_list.append(args)

    #args_list =[args_list[i] for i in [1]]
    #f1_scores =[f1_scores[i] for i in [1]]
    ensemble_method ='weighted_avg' 
    #ensemble_method = 'majority_voting'
    main(args_list, f1_scores=f1_scores, ensemble_method=ensemble_method) # majority_voting
