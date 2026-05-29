import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else (torch.device("mps") if torch.mps.is_available() else torch.device("cpu"))
    
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True, pin_memory=True) # pin_memory для ускорения копирования

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        noise = torch.randn(data.shape, device=device) # device для создания на гпу
        data = data.to('cuda', non_blocking=True) # non_blocking
        target = target.to('cuda', non_blocking=True)
        
        data = data + noise

        optimizer.zero_grad(set_to_none=True) # быстрее + меньше памяти

        # cuda работает асинхронно поэтому время исполнения нужно считать учитывая это
        time_start = torch.cuda.Event(enable_timing=True) 
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()
        output = model(data)
        loss = criterion(output, target)
        forward_times.append((time_start, time_end))

        time_start_bwd = torch.cuda.Event(enable_timing=True)
        time_end_bwd = torch.cuda.Event(enable_timing=True)
        time_start_bwd.record()
        loss.backward()
        time_end_bwd.record()
        backward_times.append((time_start_bwd, time_end_bwd))

        optimizer.step()

        losses_history.append(loss.detach()) # чтобы не засорять гпу
        print(f"Batch {batch_idx} loss: {loss.detach().cpu().item():.4f}") # loss.item() вызывает синхронизацию гпу с цпу (мешает нормальному async loop)
        
        # torch.cuda.empty_cache() это ваще не надо

    torch.cuda.synchronize()
    forward_times = [start.elapsed_time(end) for start, end in forward_times]
    backward_times = [start.elapsed_time(end) for start, end in backward_times]

    

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()