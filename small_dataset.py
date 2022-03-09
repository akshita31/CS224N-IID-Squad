import numpy as np

data = np.load('data/train.npz')

np.savez('data/smaller_train.npz', context_idxs=data['context_idxs'][:10], 
                                    context_char_idxs=data['context_char_idxs'][:10], 
                                    ques_idxs=data['ques_idxs'][:10], 
                                    ques_char_idxs=data['ques_char_idxs'][:10], 
                                    y1s=data['y1s'][:10], 
                                    y2s=data['y2s'][:10], 
                                    ids=data['ids'][:10])