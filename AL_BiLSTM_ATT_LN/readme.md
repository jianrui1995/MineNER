### 问题  
* 如何让参数不训练 layer.trainable=False，没有用上这个技术。还是用上了逻辑判断语句。
* 训练过程，验证过程，（样本选择过程）单独拎出来。（ 判断逻辑） √
    * 训练过程，要贴上LL，callback控制是否启动LL，直接用fit。所以在训练过程中不能用training来设计，使用一个标志符号，isUseLL√
    * 验证过程，样本的validation，使用training=False，使得LL不起作用。√
    * 样本选择过程，使用predict(方法)，在callback中单独修改tranining为Ture，model中内置一个ispredict参数，确定你是不是要用来预测loss。
* 贴Loss Learning。√
* 数据集准备 √
* LL中，输出的dense加不及激活函数？ 要加，而且论文中只有一层，打算写两层dense。√
* LL的损失计算 √
* 对loss类进行补充 √
* 随机生成初始化的数据集500条用来初始化的训练。√
* callback补全,停用LL
* 训练的方法√
* 在存储后，注意修改人为设置的变量
### Loss Learning的拟合算法。
1 每一层的输出做平均池化，然后放入全连接层。两层，最后输出一个点。激活函数用relu。  
2 使用卷积。


