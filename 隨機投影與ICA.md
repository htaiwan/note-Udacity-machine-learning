# 隨機投影與ICA

## Agenda

- [隨機投影](#1)
- [獨立成分分析ICA](#2)
- [獨立成分分析算法](#3)
- [sklearn中的ICA](#4)
- [[Lab]獨立成分分析](#5)
- [ICA應用](#6)

## Note

<h2 id="1">隨機投影</h2>

- **Random Projection**: 主要還是處理資料的降維。
- 跟PCA的差異: PCA會找出最大方差的方向才開始進行投影降維的動作，在高維資料時這樣的計算是很耗效能的，random projection就是隨便找個方向就進行投影降維動作，雖然效能較好，但資料有可能失真。

![456](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/456.png)

- 從數學角度: 就是原本數據乘上一個隨機矩陣。

![457](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/457.png)

- eps: 超參數，可容許的誤差(在sklearn中預設是0.1)，透過指定的誤差，sklearn會自動幫我們計算找出最佳的維度空間(12000 -> 6268)。

![458](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/458.png)

![459](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/459.png)

![460](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/460.png)


<h2 id="2">獨立成分分析ICA</h2>

- **Independent Component Analysis**: 
	- 雞尾酒會問題: 如何將合成聲音分離出各自獨立的音源。
	- input: 3組不同的合成音(每組合成音是3個獨立componet合成)
	- output: 3組獨立的音源。

![461](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/461.png)

<h2 id="3">獨立成分分析算法</h2>

- X = A * S
	- X: 合成音
	- S: 獨立音
	- A: 混合矩陣
	- 目標: 由X找出S

![462](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/462.png)

- S = W * X
	- W: 非混合矩陣
	- 目標: ICA目標是找出最佳的W，然後透過Ｗ來得出S

![463](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/463.png)

- [独立成分分析：算法与应用](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/10.1.1.322.679.pdf)
	- FastICA: 兩個重要假設
		- 訊號源必須是獨立。
		- 訊號源必須是非高斯分佈。(如果是高斯分佈就沒辦法恢復成原始信號)
	
![464](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/464.png)

<h2 id="4">sklearn中的ICA</h2>

![465](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/465.png)

<h2 id="5">[Lab]獨立成分分析</h2>

- [獨立成分分析](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Jupyter/Independent_Component_%20Analysis_Lab-zh.ipynb)

<h2 id="6">ICA應用</h2>

![466](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/466.png)

![467](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/467.png)

![468](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/468.png)