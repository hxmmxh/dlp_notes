
# Dataset类
-读入数据集数据并且对读入的数据进行索引
```
torch.utils.data.Dataset
```
- 所有自定义的dateset都必须继承自Dataset
- 必须重写三个方法:`__init__(), __getitem__(), __len__()`
  - 其中__len__应该返回数据集的大小
  - 而__getitem__应该编写支持数据集索引的函数，例如通过dataset[i]可以得到数据集中的第i+1个数据。

```py
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self,dir_root):
        ...
    
    def __getitem__(self, item):
        ...

    def __len()__(self):
        ...    
```

# IterableDataset类
- 可以用于流式读取文件夹，文件夹下所有大数据文件，逐个文件
- 必须重写`__iter__()`
```py
from torch.utils.data import IterableDataset


```





## DataLoader类
- 对于DataLoader类，如果要自定义，则一般需要完成__init__和__len__方法。如果无需更多配置，则将自定义的Dataset类传入DataLoader即可
- https://blog.csdn.net/weixin_35757704/article/details/119715900  

```py
from torch.util.data import DataLoader
from torch.utils.data import IterableDataset, DataLoader
import glob

class MyIterableDataset(IterableDataset):

    def __init__(self, file_list):
        super(MyIterableDataset, self).__init__()
        self.file_list = file_list

    def parse_file(self):
        for file in self.file_list:
            print("读取文件：", file)
            with open(file, 'r') as file_obj:
                for line in file_obj:
                    yield line

    def __iter__(self):
        return self.parse_file()


if __name__ == '__main__':
    all_file_list = glob.glob("datas/*.csv")  # 得到datas目录下的所有csv文件的路径
    dataset = MyIterableDataset(all_file_list)

    # 这里batch_size=3，意味着每次读取dataloader都会循环三次dataset
    # drop_last是指到最后，如果凑够了3个数据就返回，如果凑不够就舍弃掉最后的数据
    dataloader = DataLoader(dataset, batch_size=3, drop_last=True)
    for data in dataloader:
        print(data)

```

- DataLoader类参数详解
```
dataset (Dataset): dataset from which to load the data. 
自定义的Dataset

batch_size (int, optional): 每次都DataLoader会循环多个次dataset 
mini batch的大小，通常把batch_size改大一点，为2的整数次幂

shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``). 
在每轮训练后，将数据集打乱

sampler (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset. Can be any ``Iterable`` with ``__len__`` implemented. If specified, :attr:`shuffle` must not be specified.
自定义方法（某种顺序）从Dataset中取样本，指定这个参数就不能设置shuffle
指定shuffle相当于使用内置的RandomSampler进行采样，否则使用SequentialSampler
RandomSampler的__iter__方法有一行代码：yield from torch.randperm(n, generator=self.generator).tolist()
SequentialSampler: return iter(range(len(self.data_source)))，均继承了Sampler[int]

batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but returns a batch of indices at a time. Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`. 
返回一个batch的索引，与batch_size, shuffle, sampler, drop_last互斥
传入了batch_sampler，相当于已经告诉了PyTorch如何从Dataset取多少数据，怎么取数据去组成一个mini batch，所以不需要以上参数

num_workers (int, optional): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default: ``0``) 
多进程加载数据，默认为0，即采用主进程加载数据

collate_fn (callable, optional): merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset. 
聚集函数，用来对一个batch进行后处理，拿到一个batch的数据后进行什么处理，用这个参数定义，返回处理后的batch数据
常用默认：_utils.collate.default_collate，源码中进行了若干逻辑判断，仅将数据组合起来返回，没有实质性工作
默认collate_fn的声明是：def default_collate(batch): 所以自定义collate_fn需要以batch为输入，以处理后的batch为输出

pin_memory (bool, optional): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.  If your data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type, see the example below.
用于将tensor加载到GPU中进行运算

drop_last (bool, optional): set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: ``False``)
是否保存最后一个mini batch，样本数量可能不支持被batch size整除，所以drop_last参数决定是否保留最后一个可能批量较小的batch

timeout (numeric, optional): if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: ``0``)
控制从进程中获取一个batch数据的时延

worker_init_fn (callable, optional): If not ``None``, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``None``)
初始化子进程

prefetch_factor (int, optional, keyword-only arg): Number of samples loaded in advance by each worker. ``2`` means there will be a total of 2 * num_workers samples prefetched across all workers. (default: ``2``)
控制样本在每个进程里的预加载，默认为2

persistent_workers (bool, optional): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers `Dataset` instances alive. (default: ``False``)
控制加载完一次Dataset是否保留进程，默认为False

```