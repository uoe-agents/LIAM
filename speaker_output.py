import torch

def speaker_output(agent_id, input):
    values, color = input[0].max(0)
    color = int(color.data)
    if agent_id == 1:
        if color == 0:
            return torch.Tensor([[1, 0, 0, 0, 0]])
        if color == 1:
            return torch.Tensor([[0, 1, 0, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 1, 0, 0]])

    if agent_id == 2:
        if color == 0:
            return torch.Tensor([[0, 1, 0, 0, 0]])
        if color == 1:
            return torch.Tensor([[0, 0, 1, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 0, 1, 0]])

    if agent_id == 3:
        if color == 0:
            return torch.Tensor([[0, 1, 0, 0, 0]])
        if color == 1:
            return torch.Tensor([[1, 0, 0, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 0, 1, 0]])

    if agent_id == 4:
        if color == 0:
            return torch.Tensor([[0, 0, 0, 1, 0]])
        if color == 1:
            return torch.Tensor([[1, 0, 0, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 0, 0, 1]])

    if agent_id == 5:
        if color == 0:
            return torch.Tensor([[1, 0, 0, 0, 0]])
        if color == 1:
            return torch.Tensor([[0, 0, 1, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 0, 0, 1]])

    if agent_id == 6:
        if color == 0:
            return torch.Tensor([[0, 0, 0, 0, 1]])
        if color == 1:
            return torch.Tensor([[0, 0, 0, 1, 0]])
        if color == 2:
            return torch.Tensor([[0, 0, 1, 0, 0]])

    if agent_id == 7:
        if color == 0:
            return torch.Tensor([[0, 0, 0, 1, 0]])
        if color == 1:
            return torch.Tensor([[0, 0, 1, 0, 0]])
        if color == 2:
            return torch.Tensor([[0, 1, 0, 0, 0]])

    if agent_id == 8:
        if color == 0:
            return torch.Tensor([[0, 0, 1, 0, 0]])
        if color == 1:
            return torch.Tensor([[0, 0, 0, 0, 1]])
        if color == 2:
            return torch.Tensor([[1, 0, 0, 0, 0]])

    if agent_id == 9:
        if color == 0:
            return torch.Tensor([[0, 0, 0, 1, 0]])
        if color == 1:
            return torch.Tensor([[0, 0, 0, 0, 1]])
        if color == 2:
            return torch.Tensor([[0, 1, 0, 0, 0]])

    if agent_id == 10:
        if color == 0:
            return torch.Tensor([[0, 0, 1, 0, 0]])
        if color == 1:
            return torch.Tensor([[0, 1, 0, 0, 0]])
        if color == 2:
            return torch.Tensor([[1, 0, 0, 0, 0]])
