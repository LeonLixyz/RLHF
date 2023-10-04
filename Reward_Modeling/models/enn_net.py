import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
import torch.nn.functional as F

class Epinet(nn.Module):
    def __init__(self, hid_rep_size, ref_size, hidden_size, output_size, gain, lmbda = 1.0):
        super(Epinet, self).__init__()
        self.ref_size = ref_size
        self.output_size = output_size
        self.gain = gain
        self.lmbda = lmbda

        self.model = nn.Sequential(
            nn.Linear(hid_rep_size + ref_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, ref_size * output_size),
        )

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=self.gain)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, z, return_full_z=False):
        # stop gradient for x_tilda:
        x = x.detach() # Detach x here to treat sg[phi_zeta(x)] as a constant
        # x has shape (batch_size, num_Z_size, hid_rep_size)
        # z has shape (num_Z_size, ref_size)


        batch_size = x.shape[0]
        num_Z_size = x.shape[1]
        assert num_Z_size == z.shape[0], "wrong z dimension"
        # repeat z at 0th dimension to match batch_size
        z = z.unsqueeze(0).repeat(batch_size, 1, 1)

        x_tilda = torch.cat((x, z), dim=2)
        
        x_tilda = self.model(x_tilda)
        # Reshape the output to (batch_size, ref_size, output_size)
        x_tilda = x_tilda.view(batch_size, num_Z_size, self.ref_size, self.output_size)

        # xtilda transpose @ z
        x_tilda = torch.transpose(x_tilda, 2, 3) @ z.unsqueeze(3)
        if return_full_z:
            return self.lmbda * x_tilda.squeeze()
        else:
            x_tilda = torch.mean(x_tilda, dim=1)
            x_tilda = x_tilda.view(-1, self.output_size)
            return self.lmbda * x_tilda
    
