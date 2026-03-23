import subprocess
import torch

def get_available_cuda_devices(min_memory_gb=20, max_memory_usage=5, max_gpu_util=15):
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')

        lines = result.strip().split('\n')
        available_devices = []

        for i, line in enumerate(lines):
            total_mem, used_mem, gpu_util = map(int, line.split(','))
            total_gb = total_mem / 1024
            used_gb = used_mem / 1024
            mem_usage = used_gb / total_gb * 100

            if total_gb >= min_memory_gb and mem_usage < max_memory_usage and gpu_util < max_gpu_util:
                available_devices.append(i) #torch.device(f'cuda:{i}').item()

        return available_devices,len(lines)

    except Exception as e:
        print("无法调用 nvidia-smi，请确认已安装 NVIDIA 驱动且 nvidia-smi 可用。")
        print("错误信息：", e)
        return []
