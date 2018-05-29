## 深度学习在 CTR 方面的应用

### 背景

计算广告要解决的问题是为用户找到最匹配的广告，使得广告平台的 eCPM 期望最大。在具体投放广告的过程中会对将要推送给用户的广告进行排序，排序是按照投放给用户的广告的期望收益 `eCPM = bid * p_ctr` 大小排序。这里`bid`指的是广告主的对单次点击的出价，`p_ctr` 是预估点击率。所以，点击率预估地准不准，将直接影响到广告平台的收益。

最广泛使用的点击率预估模型是Logistic Regression(LR) 模型，其假设函数为：

```math
y_{LR}(x) = sigmoid(\sum_{i=1}^Nw_{i}x_{i} + w_{0})
```
LR 的优势在于其简单、高效，以及很好的可解释性，要学习的参数 `$w_i$` 就是特征 `$x_i$` 对于最终预测值的贡献程度。但 LR 本质上还是一个线性模型，需要大量的特征工程才能学习到高阶的交叉特征。

最近这几年，深度学习技术凭借着其强大的表达能力以及灵活的网络结构在图像、语音以及自然语言处理领域等众多领域大放异彩，取得了令人瞩目的突破。在广告推荐领域，许多公司开始尝试将深度学习的技术应用到点击率预测当中，本文也将对几种深度模型进行简单介绍。

在进行介绍前，说明下各个变量的定义：

- `$n$`: 特征总数，也就是对特征 One-hot 后的向量总长度
- `$f$`: 特征域个数，表示有多少个特征类别
- `$e$`: embedding 向量维度
- `$H_l$`: 第 `$l$` 层隐藏层单元个数

### 深度神经网络(DNN)

广告领域对于深度学习的应用比较简单，一般不会用到特别复杂的网络结构，层数一般 3-5 层。下图是一个由输入层、输出层以及2个隐藏层构成的 4 层DNN 网络，各层之间通过全连接相连。

![mlp](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEBe3869c847df2d37082052d9d059001e4&version=7944&cstk=vmTFX9BV)

其中第 `$l+1$` 层网络的输出 `$a^{l+1}$` 为：

```math
a^{l+1} = \sigma(W^{l}a^{l} + b^{l})
```

`$W^l$`为 `$l$` 层系数矩阵， `$a^{l}$` 为 `$l$` 层输出，`$b^{l}$` 为 `$l$` 层偏置项，`$\sigma$` 为激活函数，一般是 sigmoid 函数。

DNN 的训练通过反向传播算法进行，可使用 SGD，Adagrad，Adam 等常见优化算法进行参数更新，正则化可通过设置 dropout 或者 l2 。整个 DNN 需要学习的参数量: `$n * H_1 + H_1 * H_2 + H_2 * 1$` 

LR 模型也可以看做是一个只有输入输出层，没有隐藏层的两层神经网络，其输入直接通过sigmoid函数激活输出。如下图： 

![lr](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEB077049bcfa745612ee642f285db9a4fc&version=8185&cstk=vmTFX9BV)

在广告领域，大部分的特征都是类别特征(categorical feature)，例如广告 ID，网络类型特征。对于连续值特征，有时候也会利用分桶的方式将其转化成类别特征。针对类别特征，一般采用 One-hot 编码，例如网络类型有三种：3G，4G，WIFI。One-hot 编码后向量长度为3，分别为[1,0,0]，[0,1,0]，[0,0,1]。

直接将特征 One-hot 编码后放到 DNN 里，会使得输入层变得非常稀疏，且维度变得很大，这会使得要学习的参数量会变得非常大，在现实场景中是不可接受的。深度神经网络的输入一般是稠密的实数值向量，如何将高维稀疏向量转化成低维稠密的实数值向量，其实CTR 预估领域经常用到的FM模型已经给出了解决方案。

```math
y_{FM} = sigmoid(w_{0} + \sum_{i=1}^Nw_{i}x_{i} + \sum_{i=1}^N\sum_{j=i+1}^N<V_{i}, V_{j}>x_{i}x_{j})
```

FM 模型由 LR 部分以及特征交叉部分组成，FM 的优势在于求 `$x_{i}$` 与 `$x_{j}$` 交叉特征的权重时，不再像以前一样需要 `$x_{i}$` 与 `$x_{j}$` 同时存在于训练样本中，而是对每个特征 `$x_{i}$` 都会学习到一个长度为 `$e$` （量级在O(1), O(10)）的实数值隐向量 `$V_{i}$`，交叉特征的权重等于特征对应隐向量的内积。

FM 模型也可以看做是如下的前向神经网络，各层之间不再是全连接。

![fm](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEB09577cb8f0f16f6e4cdc8febca3f1035&version=7953&cstk=vmTFX9BV)

最下面是特征维度为 `$n$` 的离散输入，embedding 层可以看做是 DNN 的输入层，从而将 DNN 的输入维度从 `$n$` 变成了 `$f*e$`，其值要远远小于 `$n$`。经过 embedding 层后，对线性部分(LR)和非线性部分（特征交叉）累加后经过sigmoid输出。

对于 FM，需要学习的参数：

- LR 部分：`$n+1$`
- 特征交叉部分：`$n * e$`

### Wide & Deep 模型

谷歌在 2016 年提出了 Wide & Deep 模型，并将其用于 Google Play App 推荐，取得了不错的效果。

![wide&deep](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEBd3a5d88ce82d8643e5c6ac7fed25944f&version=7831&cstk=vmTFX9BV)

其中 Deep 部分是 DNN，原始特征经过 Embedding 后将各个 embedding 向量拼接起来构成 DNN 的输入 `$a^0=[V_1, V_2,...,V_f]$`; Wide 部分由LR以及交叉特征 `$\varPhi(x)$` 构成，其中 `$\varPhi(x)$` 表达了两个特征同时出现在样本中对于最终预测值的贡献程度，往往需要做大量特征工程工作。最终两者的输出累加后经过 sigmoid 激活输出。

```math
y_{w\&d} = sigmoid(y_{wide} + y_{deep})

y_{wide}(x) = w_{wide}^T[x, \varPhi(x)]+b

y_{deep}(x) = \sigma(W_{deep}^{T}a^{l} + b^{l})

```

至于为什么这么设计，谷歌在论文中提到两个重要概念，记忆性以及泛化性。
- 记忆性：记忆性是指学习在训练数据中高频特征与标签之间关联的能力。广义线性模型比如 LR 的优势在于记忆性好，对于样本中出现的高频低阶特征能够用少量参数学习，缺点在于泛化能力差，对于没见过的 ID 类特征，模型学习能力比较差。
- 泛化性是指学习在历史数据中低频特征与标签之间关联的能力。基于隐向量的 DNN 模型优势在于其泛化性好，对于样本中少量出现甚至没出现过的样本都能做出预测。

Wide & Deep 模型通过联合训练 Wide & Deep，累加 Wide & Deep 的结果，结合了广义线性模型记忆性好以及 DNN 模型泛化性好的优点，在 Google app store 的推荐上取得了不错的效果。该模型需要学习的参数：

- Wide: `$n+1$`
- Deep: `$f*e*H_1 + H_1*H_2+H_2*1$` 
- Embedding: `$n * e$`

谷歌提出的 Wide & Deep 框架基本成为了“业界标准”，在它之后的许多模型通过变换 Wide 端或者 Deep 端的组件形成新的模型。

### DeepFM 模型

由于 Wide & Deep 模型中的 Wide 端 LR 还需要特征工程去设计交叉特征，华为以及哈工大实验室随后提出了DeepFM 模型。DeepFM 将 Wide 端的 LR 替换为 FM，省去了 LR 特征工程的工作，而且由于 Wide & Deep 共享同一个 Embedding 层，训练也会变得更加高效。

![deepfm](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEB432731fc876e263a8777df3d9a0e62bc&version=7852&cstk=vmTFX9BV)

左边是 FM 层，用来学习 order-1 以及 order-2 交叉特征。右边是 DNN 层，用来学习更高阶的交叉特征。

```math
y_{deepfm} = sigmoid(y_{fm} + y_{dnn})

y_{fm} = w_{0} + \sum_{i=1}^Nw_{i}x_{i} + \sum_{i=1}^N\sum_{j=i+1}^N<V_{i}, V_{j}>x_{i}x_{j}

y_{dnn} = \sigma(W^la + b)
```

DeepFM 需要学习的参数:

- Deep: `$n*e*H_1 + H_1*H_2 + H_2*1$` 
- FM: `$1+n+n*e$`

### Neural Factorization Machine(NFM)

NFM 模型用 DNN 来改进 FM 的交叉特征部分的学习。

```math
y_{nfm}(x) = w_{0} + \sum_{i=1}^Nw_{i}x_{i}+f(x)
```

`$f(x)$` 的结构如下所示:
![nfm](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEB4f28869777b6fc4cb7d73236861ffcc9&version=7979&cstk=vmTFX9BV)

原始特征经过 Embedding 后拼成 `$V_{x}={x_{i}v_{i}}, x_{i} \neq 0$`，经过一个 Bi-Interaction Pooling(BI) 后作为 DNN 的输入。BI 层的操作如下：

```math
f_{BI}(V_x) = \sum_{i=1}^N\sum_{j=i+1}^NV_i \odot V_j
```
`$\odot$` 代表两个向量对应元素进行乘积。经过 BI 操作后，DNN 输入的维度从 `$n$` 降低到 `$e$`（一般设置为O(10),O(100)），降低了网络复杂度，从而提高了模型训练的效率。但与此同时，这种方式可能会带来比较大的信息损失。

该模型需要学习的参数：

- LR: `$n+1$`
- NFM: `$n*e+e*H_1+H_1*H_2 + H_2*1$`

### Deep&Cross Network (DCN)

![dcn](https://note.youdao.com/yws/api/personal/sync?method=download&fileId=WEBf6e228e48bbf4d018185632d0a61351c&version=7978&cstk=vmTFX9BV)

DCN 模型由 Deep 端和 Cross 端构成，Deep 是一个典型的 DNN 网络，Cross 端构造地很巧妙，可以高效地学习高阶的交叉特征，公式如下： 

```math
x_{l+1}=x_0x_{l}^Tw_l + b_l + x_l

x_0=[v_1, v_2, ..., v_{f}]
```
其中 `$v_i$` 是特征 `$xi$` 对应的 embeding 向量，在 DCN 中，embedding 向量的维度是不固定的，论文中对于 criteo 数据集取值 `$6(category \; cardinality)^{\frac{1}{4}}$`。`$x_0$` 由 `$v_i$` 拼接而成，假设 `$x_0$` 维度是 `$d$`，
`$x_l$` 能够表达最高 `$l+1$` 次的特征交叉，但是这一层需要学习的参数只有`$w_l$` 和 `$b_l$`，维度与 `$x_0$` 相同， 总共 2 * `$d$` 个，所以 Cross 层可以高效地学习高阶交叉特征。

模型需要学习的参数:

- Embedding: 固定 embedding 向量维度为 `$e$`，则为 `$n*e$`
- Deep: `$d$` * `$H1$` + `$H1$` * `$H2$` + `$H2$` * 1
- Cross: 2 * `$d$` * `$L$`, `$L$` 是Cross的层数

### 参考:

1. 
Wide & Deep Learning for Recommender Systems: https://arxiv.org/abs/1606.07792.pdf
2. 
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction: https://arxiv.org/abs/1703.04247.pdf
3. 
Neural Factorization Machines for Sparse Predictive Analytics: https://arxiv.org/abs/1708.05027.pdf

4.Deep & Cross Network for Ad Click Predictions: https://arxiv.org/pdf/1708.05123.pdf