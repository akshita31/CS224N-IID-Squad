import numpy as np

data = np.load('data/train.npz')

np.savez('data/smaller_train-25.npz', context_idxs=data['context_idxs'][:25], 
                                    context_char_idxs=data['context_char_idxs'][:25], 
                                    ques_idxs=data['ques_idxs'][:25], 
                                    ques_char_idxs=data['ques_char_idxs'][:25], 
                                    y1s=data['y1s'][:25], 
                                    y2s=data['y2s'][:25], 
                                    ids=data['ids'][:25])