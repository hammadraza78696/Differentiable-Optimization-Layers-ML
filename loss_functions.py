import torch
import torch.nn as nn

class ConstrainedLossObjective(nn.Module):
    """
    Custom loss module penalizing parameter drift away from 
    feasible mathematical budget space barriers.
    """
    def __init__(self, penalty_coefficient=0.5):
        super(ConstrainedLossObjective, self).__init__()
        self.gamma = penalty_coefficient
        self.base_loss = nn.MSELoss()
        
    def forward(self, allocations, targets, budget=1.0):
        mse = self.base_loss(allocations, targets)
        
        # Mathematical penalty for budget violation: (Sum(y) - B)^2
        budget_violation = torch.mean((torch.sum(allocations, dim=-1) - budget) ** 2)
        
        total_loss = mse + (self.gamma * budget_violation)
        return total_loss
