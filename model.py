import argparse
import string
import numpy as np
# 导入keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
# import tushare as ts
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

ap = argparse.ArgumentParser(description='Process some integers.')
ap.add_argument("--action", default="predict", help="train or predict")
ap.add_argument("--path", default="./chk", help="path to save or load model")
ap.add_argument("--epochs", type=int, default=100, help="epochs for train")
ap.add_argument("--batch", type=int, default=64, help="batch size for train")
ap.add_argument("--count", type=int, default=100, help="sample data count")
ap.add_argument("--show", type=bool, default=False, help="whether show result in chart")


def get_beans4(counts):
    # 生成相应的数据函数
    xs = np.random.rand(counts, 2)*2
    ys = np.zeros(counts)
    for i in range(counts):
        x = xs[i]
        if (np.power(x[0]-1, 2)+np.power(x[1]-0.3, 2)) < 0.5:
            ys[i] = 1
    return xs, ys


def show_3d_scatter(X, Y):
    # 画3d散点图
    x = X[:, 0]
    z = X[:, 1]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, z, Y)
    plt.show()


def show_scatter(X, Y):
    # 画出数据的散点图
    if X.ndim > 1:
        show_3d_scatter(X, Y)
    else:
        plt.scatter(X, Y)
        plt.show()


def show_3d_scatter(X, Y):
    # 画3d散点图
    x = X[:, 0]
    z = X[:, 1]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, z, Y)
    plt.show()


def show_scatter_surface_with_model(X, Y, model):
    # 画3D图
    # model.predict(X)
    x = X[:, 0]
    z = X[:, 1]
    y = Y

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(x, z, y)

    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)

    X = np.column_stack((x[0], z[0]))

    for j in range(z.shape[0]):
        if j == 0:
            continue
        X = np.vstack((X, np.column_stack((x[0], z[j]))))

    y = model.predict(X)

    y = np.array([y])
    y = y.reshape(x.shape[0], z.shape[1])
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()


def model_create():
    # 建立网络模型
    model = Sequential()
    model.add(Dense(units=10, activation='sigmoid', input_dim=2))
    # units 神经元个数， activation激活函数类型， 输了特征维度
    model.add(Dense(units=1, activation='sigmoid'))  # 输出层
    # 编译网络
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.3), metrics=['accuracy'])
    # mean_squared_error 均方误差 sgd 随机梯度下降算法 accuracy 准确度
    return model


def model_train(model, EPOCHS, BATCHSIZE, X, Y):
    # 训练回合数epochs， batch_size 批数量，一次训练利用多少样本
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCHSIZE)


def model_save(model, path):
    model.save(path)
    print('save model to ', path)


def train(EPOCHS, BATCHSIZE, X, Y):
    model = model_create()
    model_train(model, EPOCHS, BATCHSIZE, X, Y)
    return model


def predict(model, X):
    # 预测函数
    pres = model.predict(X)
    print(pres)


if __name__ == '__main__':
    # python model.py --action=train --path=./chk --epochs=100 --batch=128 --count=100
    # python model.py --action=predict --path=./chk --epochs=100 --batch=128 --count=100
    # python model.py --action=predict --path=./chk --epochs=100 --batch=128 --count=100 --show=true
    args = ap.parse_args()
    print("action:", args.action)
    print("path:", args.path)
    print("epochs:", args.epochs)
    print("batch:", args.batch)
    print("count:", args.count)
    print("show:", args.show)

    # 数据量
    m = args.count
    X, Y = get_beans4(m)
    print(X)
    print(X.shape)
    # show_scatter(X, Y)
    if args.action == 'train':
        model = train(args.epochs, args.batch, X, Y)
        if args.path != '':
            model_save(model, args.path)

    if args.action == 'predict' and args.path != '':
        model = load_model(args.path)
        predict(model, X)

    # 三维的
    # if args.show == True and (args.action == 'train' or args.action == 'predict'):
    #     show_scatter_surface_with_model(X, Y, model)
