# PCA

## Agenda

- [用於數據轉換的PCA](#1)
- [PCA的中心跟新軸](#2)
- [哪些數據可用於PCA](#3)
- [軸何時佔主導位置](#4)
- [複合特徵](#5)
- [最大方差與訊息損失](#6)
- [用於特徵轉換的PCA](#7)
- [PCA的回顧與定義](#8)
- [sklearn中的PCA](#9)
- [何時使用PCA](#10)
- [用於人臉辨識的PCA](#11)

## Note

<h2 id="1">用於數據轉換的PCA</h2>

- **Pricipal Component Analysis**: PCA是將舊座標系統透過平移，旋轉的方式轉換成新座標系統。
	- X軸: 資料變化的主軸
	- Y軸: 重要性較低的變化方向。

![439](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/439.png)


<h2 id="2">PCA的中心跟新軸</h2>

- 新中心: PCA新座標的中心將是資料集分布的中心點。
- x': 當在舊座標系統下移1格(-1), 新座標系統的x平移2格(2 = 5 - 3)。
- y': 當在舊座標系統右移1格(1), 新座標系統的y上移2格(2 = 5 - 3)。

![440](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/440.png)

<h2 id="3">哪些數據可用於PCA</h2>

- 基本上，PCA可以用來分析各種不同數據的分佈。

![441](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/441.png)

<h2 id="4">軸何時佔主導位置</h2>

- 下圖(中)的數據分佈，即使轉換到新座標系統，資料分布在兩個不同軸仍佔相同範圍，此時軸就不佔主導位置。

![442](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/442.png)

<h2 id="5">複合特徵</h2>

- 將兩個獨立特徵(房間數量跟房子大小)，透過PCA轉化成一個複合特徵。
- 壓縮特徵空間維度。

![443](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/443.png)

<h2 id="6">最大方差與訊息損失</h2>

- 何謂最大方差(variance) ?
	- 從ML的角度解釋: 一個算法願意學習的程度。
	- 從統計學的角度解釋: 數據的大致分布。

![444](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/444.png)

![445](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/445.png)
	
- 最大方差的優點 ?
	- 當數據沿著最大方差的維度進行映射時，可以最大程度保留原始數據的信息量。

![446](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/446.png)

- 訊息損失 ？
	- 數據到最大方差的維度的距離。
	- 訊息損失量跟此距離此成比例的。

![447](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/447.png)

- 當將方差進行最大化時，實際上就是將點到該線投影的距離進行最小化。

![448](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/448.png)

<h2 id="7">用於特徵轉換的PCA</h2>

- 透過人工方式進行特徵組合，是不合理的，當有上萬個特徵時，是很難由人工進行特徵組合。

![449](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/449.png)

- 正確做法，將所有特徵直接丟入PCA之中，讓PCA來自動將這些特徵組合成一些新特徵。

![450](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/450.png)


<h2 id="8">PCA的回顧與定義</h2>

- PCA是將輸入特徵轉化成其主成分的系統方式。
- 這些主成份可以當作新的組合特徵。
- 主成份的定義是數據沿著最大方差的方向進行映射。
- 數據因特定主成分而產生的方差越大，那麼該主成份的級別越高。
- 這些主成份在數學角度是相互垂直的，也就是說不同的主成份間不會有重疊。
- 主成份的數量是有上限，最多就是跟原始的特徵數量相同。

![451](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/451.png)


<h2 id="9">sklearn中的PCA</h2>

![452](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/452.png)

![453](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/453.png)

<h2 id="10">何時使用PCA</h2>

- 找出data中的隱藏特徵。
- 降低維度。
	- 可視化數據。
	- 減少噪音數據(透過找出主成份)。
	- 當作數據的預處理，讓其他算法表現更好。

![454](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/454.png)

<h2 id="11">用於人臉辨識的PCA</h2>

- 為什麼PCA在人臉辨識上有不錯的效果？
	- 人臉照片通常有很高的輸入維度(很多像素)。
	- 人臉具有一些一般性形態，這些型態可以用較小的維度方式來補抓，比如眼睛數目，位置等。

![455](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/455.png)

- **Faces recognition example using eigenfaces and SVMs**
