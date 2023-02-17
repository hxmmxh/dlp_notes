

# 各类激活函数
https://zhuanlan.zhihu.com/p/203136050

https://blog.csdn.net/wulele2/article/details/117884253
https://blog.csdn.net/weixin_39529413/article/details/123071764

# 各种卷积

## pointwise conv
- 

# 自定义算子方法
https://blog.csdn.net/wholetus/article/details/125447128

- 扩展方法：通过继承 autograd.Function
- 继承 autograd.Function 的 子类 只需要 实现两个 静态方法：
  - forward ： 计算 op 的前向过程.
    - 在执行 forward 之前，Variable 参数已经被转换成了 Tensor
    - forward 的形参可以有默认参数，默认参数可以是任意 python 对象。
    - 可以返回任意多个 Tensor
    - 里面可以使用任何 python 操作，但是 return 的值必须是 Tensor
  - backward ： 计算 梯度，
    - forward 返回几个值， 这里就需要几个形参，还得外加一个 ctx。
    - forward 有几个 形参（不包含 ctx） ，backward 就得返回几个值。
    - backward 实参也是 Variable 。
    - backward 返回的得是 Variable。
- 注意事项
  - forward 和 backward 都得是 静态方法！！！！！

# 自定义算子已经在onnx中标准化
现今onnx支持的运算符，一般最新版本的支持的运算符信息会在github的onnx源码工程中的Operators.md中写出Operators.md.
如果，运算符已经被标准化，即在上边的列表中能找到，且在该版本的torch中，这个操作是一个ATen操作符，即在 torch/csrc/autograd/generated/VariableType.h能找到它的定义。
那就在torch/onnx/symbolic.py里面加上符号并且遵循下面的指令：

在 torch/onnx/symbolic.py里面定义符号。确保该功能与在ATen操作符在VariableType.h的功能相同。
第一个参数总是ONNX图形参数，参数的名字必须与 VariableType.h里的匹配，因为调度是依赖于关键字参数完成的。
参数排序不需要严格与VariableType.h匹配，首先的张量一定是输入的张量，然后是非张量参数。
在符号功能里，如果操作符已经在ONNX标准化了，我们只需要创建一个代码去表示在图形里面的ONNX操作符。
如果输入参数是一个张量，但是ONNX需要的是一个标量形式的输入，我们需要做个转化。_scalar可以帮助我们将一个张量转化为一个python标量，并且_if_scalar_type_as函数可以将python标量转化为PyTorch张量。

# 自定义算子没有在onnx中标准化
- 如果没有被标准化，也就代表torch.onnx模块下，也没有这个op的定义，是个非ATen操作符，那么符号功能需要加在相应的PyTorch函数类中。请阅读下面的指示
- 在相应的函数类中创建一个符号函数命名为symbolic。`定义一个@staticmethod的函数symbolic()`
- 第一个参数总是导出ONNX图形参数。
- 参数的名字除了第一个必须与前面的形式严格匹配。
- 输出元组大小必须与前面的形式严格匹配。
- 在符号功能中，如果操作符已经在ONNX标准化了，我们只需要创建一个代码去表示在图形里面的ONNX操作符。

## symbolic
- static method, 符号函数
- 第一个参数总是g,导出onnx图像参数
- 剩余的参数和forward的参数保持一致
- 返回值也和forworad保持一致
- 中间调用一个g.op函数