import socket
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SGECluster
import os

def Start_Client(gpu_name):


    hostname = socket.gethostname()
    n_workers = 1
    n_cores = 1

    wks2 = "wn-wks2.fe.hhi.de"
    gpu1 = "wn-gpu1.fe.hhi.de"
    gpu2 = "wn-gpu-104-01.fe.hhi.de"

    if hostname == wks2:
        path = "/data/cluster/projects/infineon-radar/daq_x-har/3_Walking_converted/recording-2020-01-28_11-31-55"
        mem = "20G"  # Allocated memory is critical. For this example it must be at least 16GB
        q = "wn-37.q"  # Check current queue status on https://hpc-management.fe.hhi.de/wn/phpqstat/

        cluster = SGECluster(n_workers=n_workers, cores=n_cores, memory=mem,
                             resource_spec=f"h_vmem={mem}", host=hostname, queue=q,
                             job_extra=["-v MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1"])
    elif hostname in (gpu1, gpu2):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name  # Check current status with nvidia-smi and pick GPU from 0-3
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=n_cores, host=hostname)
    else:
        raise ValueError(f"{hostname} is not a supported host. Please run this example on {wks}, {gpu1} or {gpu2}.")

    client = Client(cluster)
    client.wait_for_workers(n_workers=n_workers)
    print (client)

    return client