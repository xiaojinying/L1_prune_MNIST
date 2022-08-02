from model import LeNet
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

batchsize = 64
shuffle = True
epo = 10
learning_rate = 0.01


mnist_train=MNIST(root="E:/pycharm_project/data/MNIST", train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
mnist_test=MNIST(root="E:/pycharm_project/data/MNIST",train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
train_loader = DataLoader(mnist_train, batch_size=batchsize, shuffle=shuffle)
test_loader = DataLoader(mnist_test, batch_size=batchsize, shuffle=shuffle)

model=LeNet().cuda()
device = torch.device("cuda:0")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def l1_Regularization(model,lamda):
    l1=0
    for name,para in model.state_dict().items():
        if 'weight' in name:
            l1+=torch.sum(abs(para))
    return lamda*l1

def l2_Regularization(model,lamda):
    l2=0
    for name,para in model.state_dict().items():
        if 'weight' in name:
            l2+=torch.sum(pow(para,2))
    return  lamda*l2


def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs=inputs.to(device)
        target=target.to(device)


        # forward + backward + update
        outputs = model(inputs)  #输出会以[0.11,0.15,0.90,0.87]这样的形式输出，每个数代表该图像属于这个标签的概率
        loss = criterion(outputs, target)+l2_Regularization(model,0.0001)
        optimizer.zero_grad()  # 梯度初始化为零，把loss关于权重weight的导数变成0
        loss.backward()   #反向传播得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行参数更新

        # 把运行中的loss累加起来,设定以100为一个轮回，以这100内的平均loss为loss值
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        values,predicted = torch.max(outputs.data, dim=1)    #求outputs.data的行或者列最大值，dim=1是行最大值，dim=0是列最大值，values收到每行最大值，predicted会收到最大值的索引，即求最大概率的标签
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 100 == 99:  # 设定每100次输出准确率和损失
            print('[{}, {}]: loss: {} , acc: {}%'.format(epoch + 1, batch_idx + 1, running_loss / 100, 100 * running_correct / running_total))

            #进行清零
            running_loss = 0.0
            running_total = 0
            running_correct = 0   #如果不清零则会进行累加

def test(model):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)
            values, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[{} / {}]: Accuracy on test set: {}% ' .format(epoch+1, epo, 100 * acc))
    return acc

def prune(net,threshold,test_acc):
    pruned=0.0
    total=0.0
    model_test=LeNet().cuda()
    model_test.to(device)
    dict = net.state_dict()
    model_test.load_state_dict(dict)
    for name,para in dict.items():
        if 'weight' in name:
            r=abs(para)>threshold
            pruned+=torch.sum(r).item()
            total+=para.nelement()
            dict[name]=r*para
    model_test.load_state_dict(dict)
    prune_rate=pruned/total
    pruned_acc=test(model_test)
    if test_acc>pruned_acc:
        prune_rate,pruned_acc=prune(net,threshold/2,test_acc)
    else:
        print('阈值为{},剪枝率为{}，剪枝后的精确度为{}'.format(threshold,prune_rate,pruned_acc))
        return prune_rate,pruned_acc
    return prune_rate,pruned_acc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_sign=1
    if train_sign==1:
        for epoch in range(epo):
            train(epoch)
            acc=test(model)
    i,j=prune(model,0.1,acc)
    print(i)
    print(j)



