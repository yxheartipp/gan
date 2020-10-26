import numpy as np
import torch   # pytorch机器学习开源框架
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import copy
from torch.autograd.gradcheck import zero_gradients

class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        super(Net, self).__init__()
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将28*28个节点连接到300个节点上。
        self.fc1 = nn.Linear(28*28, 300)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将300个节点连接到100个节点上。
        self.fc2 = nn.Linear(300, 100)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将100个节点连接到10个节点上。
        self.fc3 = nn.Linear(100, 10)

    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输入x经过全连接3，然后更新x
        x = self.fc3(x)
        return x

# 定义数据转换格式
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28*28))])

# 导入数据，定义数据接口
# 1.root，表示mnist数据的加载的相对目录
# 2.train，表示是否加载数据库的训练集，false的时候加载测试集
# 3.download，表示是否自动下载mnist数据集
# 4.transform，表示是否需要对数据进行预处理，none为不进行预处理
traindata = torchvision.datasets.MNIST(root="./drive/My Drive/fgsm/mnist", train=True, download=True, transform=mnist_transform)
testdata  = torchvision.datasets.MNIST(root="./drive/My Drive/fgsm/mnist", train=False, download=True, transform=mnist_transform)

# 将训练集的*张图片划分成*份，每份256(batch_size)张图，用于mini-batch输入
# shffule=True在表示不同批次的数据遍历时，打乱顺序
# num_workers=n表示使用n个子进程来加载数据
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

# 展示图片
index = 100
image = testdata[index][0]
label = testdata[index][1]
image.resize_(28,28) # 调整图片大小
img = transforms.ToPILImage()(image)
plt.imshow(img)
plt.show()

index = 100
batch = iter(testloader).next() # 将testloader转换为迭代器
# 例如：如果batch_size为4，则取出来的images是4×c×h×w的向量，labels是1×4的向量
image = batch[0][index]
label = batch[1][index]
image.resize_(28,28)
img = transforms.ToPILImage()(image)
plt.imshow(img)
plt.show()

net = Net()
loss_function = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04) # 随机梯度下降优化

num_epoch = 50  
for epoch in tqdm(range(num_epoch)): # python进度条，num_epoch=50，所以每2%显示一次
    losses = 0.0
    for data in trainloader:
        inputs, labels = data # 获取输入
        # inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad() # 参数梯度置零
        # 前向+ 反向 + 优化
        outputs = net(inputs)
        loss = loss_function(outputs, labels) # 计算loss
        loss.backward() # 传回反向梯度
        optimizer.step() # 梯度传回，利用优化器将参数更新
        losses += loss.data.item() # 输出统计 
    print("*****************当前平均损失为{}*****************".format(losses/2000.0))

correct = 0 # 定义预测正确的图片数，初始化为0
total = 0 # 总共参与测试的图片数，也初始化为0
for data in testloader:
    images, labels = data 
    outputs = net(Variable(images))# 输入网络进行测试 # 因为神经网络只能输入Variable
    _, predicted = torch.max(outputs.data, 1)#返回了最大的索引，即预测出来的类别。
      # 这个_,predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
      # 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
      # torch.max(outputs.data,1) ，返回一个tuple(元组)。第二个元素是label
    total += labels.size(0) # 更新测试图片的数量
    correct += (predicted == labels).sum() # 更新正确分类的图片的数量
print("预测准确率为：{}/{}".format(correct, total))

# 保存整个网络 #...It won't be checked...是保存模型时的输出
PATH1="./drive/My Drive/fgsm/mnist_net_all.pkl"
torch.save(net,PATH1) 
# 保存网络中的参数，速度快，占空间少，PATH1是保存路径和文件名
PATH2="./drive/My Drive/fgsm/mnist_net_param.pkl"
torch.save(net.state_dict(),PATH2)
#针对上面一般的保存方法，加载的方法分别是：
#model_dict=torch.load(PATH)
#model_dict=model.load_state_dict(torch.load(PATH))

net = torch.load(PATH1) # 加载模型

index = 100 # 选择测试样本
image = testdata[index][0]
label = testdata[index][1]

outputs = net(Variable(image)) # 因为神经网络只能输入Variable
predicted = torch.max(outputs.data,0)[1]
print('预测值为：{}'.format(predicted))

image.resize_(28,28) # 显示一下测试的图片，和上文代码相同
img = transforms.ToPILImage()(image)
plt.imshow(img)
plt.show()

PATH1="./drive/My Drive/fgsm/mnist_net_all.pkl"
net = torch.load(PATH1) # 加载模型
index = 100 # 选择测试样本
image = Variable(testdata[index][0].resize_(1,784), requires_grad=True)
label = torch.tensor([testdata[index][1]])

f_image = net.forward(image).data.numpy().flatten() # flatten()函数默认是按行的方向降维
I = (np.array(f_image)).flatten().argsort()[::-1] # argsort()[::-1]表示降维排序后返回索引值
label = I[0] # 样本标签

input_shape = image.data.numpy().shape # 获取原始样本的维度，返回（第一维长度，第二维长度，...）
pert_image = copy.deepcopy(image) # 深度复制原始样本，复制出来后就独立了
w = np.zeros(input_shape) # 返回来一个给定形状和类型的用0填充的数组
r_tot = np.zeros(input_shape)

loop_i = 0 # 循环
max_iter = 50 # 最多迭代次数
overshoot = 0.0

x = Variable(pert_image, requires_grad=True) # 因为神经网络只能输入Variable
fs = net.forward(x) # 调用forward函数
fs_list = [fs[0][I[k]] for k in range(len(I))] # 每个类别的取值情况，及其对应的梯度值
k_i = label

while k_i == label and loop_i < max_iter: # 分类标签变化时结束循环
    pert = np.inf # np.inf表示一个足够大的数
    fs[0][I[0]].backward(retain_graph=True) # 反向传播，计算当前梯度；连续执行两次backward，参数表明保留backward后的中间参数。
    orig_grad = x.grad.data.numpy().copy() # 原始梯度
    
    for k in range(len(I)): # 获得x到各分类边界的距离
        zero_gradients(x)
        fs[0][I[k]].backward(retain_graph=True)
        cur_grad = x.grad.data.numpy().copy() # 现在梯度
        
        w_k = cur_grad - orig_grad
        f_k = (fs[0][I[k]] - fs[0][I[0]]).data.numpy()
        
        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
        if pert_k < pert:  # 获得最小的分类边界距离向量
            pert = pert_k # 更新perk，pert为最小距离
            w = w_k
    r_i = (pert + 1e-4) * w / np.linalg.norm(w)
    r_tot = np.float32(r_tot + r_i) # 累积扰动
    
    pert_image = image + (1+overshoot)*torch.from_numpy(r_tot) # 添加扰动
    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    k_i = np.argmax(fs.data.numpy().flatten()) # 扰动后的分类标签
    loop_i += 1
r_tot = (1+overshoot)*r_tot # 最终累积的扰动

outputs = net(pert_image.data.resize_(1,784))
predicted = torch.max(outputs.data,1)[1] #outputs含有梯度值，其处理方式与之前有所不同
print('预测值为：{}'.format(predicted[0]))

pert_image = pert_image.reshape(28,28)
img = transforms.ToPILImage()(pert_image)
plt.imshow(img)
plt.show()
