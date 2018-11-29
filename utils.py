import torch
import numpy as np

def load_part_of_model(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        print (k)

        param = src_model.get(k)
        m_dict[k].data = param


    new_model.load_state_dict(m_dict)
    return new_model