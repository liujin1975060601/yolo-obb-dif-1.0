import torch

def save_checkpoint(model_path, epoch,model,optimizer,scheduler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }
    torch.save(checkpoint, model_path)
def load_checkpoint(fname, model,optimizer,scheduler):
    # 加载 checkpoint
    checkpoint = torch.load(fname,weights_only=False)

    # 加载状态
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 读取 epoch 和 loss
    start_epoch = checkpoint.get('epoch', 0) + 1
    return start_epoch