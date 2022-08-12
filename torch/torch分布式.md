
https://www.cxyzjd.com/article/u012605037/115294898#11_ranklocal_ranknode_13
http://shomy.top/2022/01/05/torch-ddp-intro/
https://zhuanlan.zhihu.com/p/358974461
https://zhuanlan.zhihu.com/p/373395654

https://zhuanlan.zhihu.com/p/178402798
https://zhuanlan.zhihu.com/p/86441879

任务是在自己的conda环境中跑，-i指定的镜像会对任务造成影响吗。

- [基本概念](#基本概念)
- [流程](#流程)
  - [1.初始化进程组](#1初始化进程组)
  - [2.设置采样器](#2设置采样器)
  - [3.封装模型](#3封装模型)
  - [4.启动分布式训练](#4启动分布式训练)
    - [4.1 torchrun或torch.distributed.launch](#41-torchrun或torchdistributedlaunch)
    - [4.2 torch.multiprocessing.spawn](#42-torchmultiprocessingspawn)
- [几个示例](#几个示例)
  - [单机多卡](#单机多卡)


# 基本概念
- rank
  - 进程号，在多进程上下文中，我们通常假定rank 0是第一个进程或者主进程，其它进程分别具有1，2，3不同rank号，这样总共具有4个进程
- node
  - 物理节点，可以是一个容器也可以是一台机器，节点内部可以有多个GPU；nnodes指物理节点数量， nproc_per_node指每个物理节点上面进程的数量
- local_rank
  - 指在一个node上进程的相对序号，local_rank在node之间相互独立
- WORLD_SIZE
  - 全局进程总个数，即在一个分布式任务中rank的数量
- Group
  - 进程组，一个分布式任务对应了一个进程组。只有用户需要创立多个进程组时才会用到group来管理，默认情况下只有一个group


# 流程

## 1.初始化进程组
- `torch.distributed.init_process_group(backend, init_method=None, world_size=-1, rank=-1, store=None,...)`
- 尽量使用env来初始化dist？
```py
torch.distributed.init_process_group(backend, 
                                     init_method=None, 
                                     timeout=datetime.timedelta(0, 1800), 
                                     world_size=-1, 
                                     rank=-1, 
                                     store=None)
# backend ：通信后端，可选的包括：nccl（NVIDIA推出）、gloo（Facebook推出）、mpi（OpenMPI）。从测试的效果来看，如果显卡支持nccl，建议后端选择nccl，，其它硬件（非N卡）考虑用gloo、mpi（OpenMPI）
# 有三种init_method
#init_method='tcp://ip:port'： 通过指定rank 0（即：MASTER进程）的IP和端口，各个进程进行信息交换。 需指定 rank 和 world_size 这两个参数。
#init_method='file://path'：通过所有进程都可以访问共享文件系统来进行信息共享。需要指定rank和world_size参数。
#init_method=env://：从环境变量中读取分布式的信息(os.environ)，主要包括 MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE。 其中，rank和world_size可以选择手动指定，否则从环境变量读取
# 如果这两种方法都没有使用，默认使用init_method='env'的方式来初始化
```



## 2.设置采样器
- Dataloader需要把所有数据分成N份(N为worldsize), 并能正确的分发到不同的进程中，每个进程可以拿到一个数据的子集，不重叠，不交叉。这部分工作靠 DistributedSampler完成
- 首先建立一个DistributedSampler完成对象
```py
torch.utils.data.distributed.DistributedSampler(
				dataset,
				num_replicas=None, 
				rank=None, 
				shuffle=True, 
				seed=0, 
				drop_last=False)
# dataset: 需要加载的完整数据集
# num_replicas： 把数据集分成多少份，默认是当前dist的world_size
# rank: 当前进程的id，默认dist的rank
# shuffle：是否打乱
# drop_last: 如果数据长度不能被world_size整除，可以考虑是否将剩下的扔掉
# seed：随机数种子。这里需要注意，从源码中可以看出，真正的种子其实是 self.seed+self.epoch 这样的好处是，不同的epoch每个进程拿到的数据是不一样，因此需要在每个epoch开始前设置下：sampler.set_epoch(epoch)

```
- 随后传入一个DataLoader中


    
## 3.封装模型
- 将单机模型使用torch.nn.parallel.DistributedDataParallel 进行封装



## 4.启动分布式训练
- 有两种启动方式
### 4.1 torchrun或torch.distributed.launch
- 用终端命令torchrun启动
  
```
python3 -m torch.distributed.launch --配置 train.py --args参数

--nnodes: 使用的机器数量，单机的话，就默认是1了
--nproc_per_node: 单机的进程数，即单机的worldsize
--master_addr/port: 使用的主进程rank0的地址和端口
--node_rank: 当前的进程rank

#torch1.10之后可以使用torchrun，会自动使用环境变量配置各类参数，如RANK,LOCAL_RANK, WORLD_SIZE
torchrun --配置 test.py --args参数
```

### 4.2 torch.multiprocessing.spawn
```py
mp.spawn(fn, args=(), nprocs=1, join=True, daemon=False)
#fn: 进程的入口函数，该函数的第一个参数会被默认自动加入当前进程的rank， 即实际调用： fn(rank, *args)
#nprocs: 进程数量，即：world_size
#args: 函数fn的其他常规参数以tuple的形式传递
```

# 几个示例
## 单机多卡


