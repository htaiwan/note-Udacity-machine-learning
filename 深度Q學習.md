# 深度Q學習

## Agenda

- [深度Q學習簡介](#1)
- [神經網路作為值函數](#2)
- [蒙地卡羅學習](#3)
- [時間差分學習](#4)
- [Q學習](#5)
- [深度Q網路](#6)
- [經驗回放](#7)
- [固定Q目標](#8)
- [深度Q學習算法](#9)
- [DQN改進](#10)
- [實現深度Q學習](#11)
- [TensorFlow實現](#12)
- [總結](#13)


## Note

<h2 id="1">深度Q學習簡介</h2>

- 首先，如何使用神經網路當作值函數。
- 再來，調整兩大主要免模型的方法。
	- 蒙地卡羅學習
	- 時間差分學習
- Q學習是TD學習一個變形
	- 如何自己實現一個深度Q學習算法
- 利用深度學習(包括CNN或RNN)，指導AI從頭學習任務。

![685](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/685.png)

<h2 id="2">神經網路作為值函數</h2>

- 狀態值函數是將任何狀態S對應到一個實數。
	- 表示當前策略pi對該狀態的重要性。
	- 如果利用神經網路來估算該函數，則需要以向量的形式來提供。
		- 可以透過使用特徵轉換X(S)來完成向量的轉換。
		- 因此重點就轉移到要如何學習這些參數w。
		

![686](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/686.png)

- 如果有預言家可以提供正確V(S)，那就可以透過loss的定義，利用梯度下降法進行反向傳遞來更新權重。

![687](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/687.png)

- 動作值函數跟狀態值函數很相似，但都有個相同問題，沒有預言家可以告訴我們正確的值函數是多少。

![688](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/688.png)
	

<h2 id="3">蒙地卡羅學習</h2>

- 回憶下，在傳統的蒙地卡羅學習更新值函數的遞增步驟。
	- Gt是在時間t之後所收到的累積折扣獎勵。
	- 將上節所遇到問題，未知的真正值函數替換成這個Gt。
	- 如此一來，我們神經網路就可以開始運作
		- 可以開始計算loss，梯度下降，反向傳播更新。

![689](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/689.png)

- 針對動作值函數也是執行相同的步驟。

![690](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/690.png)

**函數逼近的蒙地卡羅學習**

- 首先先隨機初始化參數w。
- 根據貪婪策略定義一個策略。
- 然後不段重複下列兩個步驟，直到w開始收斂，形成最優的值函數和相應的策略。
- 通常包含一個評估步驟(Evaluation): 估算當前策略下的每個(狀態-動作)組合的值。
	- 首先，利用此策略生成一個階段。
	- 然後在該階段的每個時間步t
		- 使用(狀態-動作)組合
		- 對St,At
		- 和根據該階段剩餘時間步計算的Gt
		- 來更新參數向量w
- 接下來是完善步驟(Improvement): 根據這些q值提取epsilon貪婪策略。

![691](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/691.png)


<h2 id="4">時間差分學習</h2>

- 時間差分法是如何替換真正的值函數。
	- 在蒙地卡羅，是利用Gt來替換。
	- 在時間差分法，是利用估算回報，也是TD來替換。
		- 最簡單的TD(0)，即使使用下個獎勵和下個狀態的折扣值。

![692](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/692.png)

- 利用TD來替換未知的真值函數，這樣就可以使用具體的數據，而不是想像的數據。

![693](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/693.png)

![694](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/694.png)

- 針對動作值函數也是執行相同的步驟。

![695](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/695.png)

**函數逼近的TD(0)學習(階段性任務)**

- 首先先隨機初始化參數w。
- 根據貪婪策略定義一個策略。
- 開始重複很多個階段
	- 在每個階段都是從環境獲得的初始狀態S開始
	- 執行一個動作A並獲得獎勵R和下個狀態S'
	- 根據epsilon貪婪策略從狀態S'選擇另外一個動作A'
	- 將S,A,R,S',A'帶入梯度下降更新規則，並調整對應的權重。
	- 最後直接將S'替換成新的S，將A'替換成新的A

![696](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/696.png)

**函數逼近的TD(0)學習(連續性任務)**

![697](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/697.png)

- SARSA是一種異同策略算法，表示我們的更新策略和執行動作遵守的策略一樣，通常效果很好可以快速收斂。因為會根據最新的策略來執行動作，但是也有缺點，那就是主要的學習策略跟執行動作遵守的策略太緊密，有可能會導致只學習到局部最佳的策略，如果想要有更佳探索性的策略，那就需要**離線策略算法**。


<h2 id="5">Q學習</h2>

- Q學習是一種TD學習的離線策略的變形。
- SARSA和Q学习都是TD方法，它们都有一个缺点，即使用非线性算法逼近时，可能无法收敛于全局最优。

**函數逼近的Q學習(階段性任務)**

- 跟SARSA主要的差別在更新步驟
	- 不再根據相同的epsilon貪婪策略選擇下一個動作。
	- 而是貪婪的選擇一個動作，該動作將最大化後續的預期值。
		- 注意: 並非實際採取該動作，而只是用於執行更新步驟。
		- 實際上，我們並不需要選擇該動作，我們可以在下個狀態使用最大的Q值。
		- 因此Q學習被視為離線策略方法。
			- 根據一個策略選擇動作，根據外一個策略進行學習。 

![698](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/698.png)

**函數逼近的Q學習(連續性任務)**

![699](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/699.png)

**SRASA VS Q-Learning**

- SARSA:
	- "遵守"跟"學習"是根據同一個策略。
	- 適合在線學習。
	- 但如果想要進行更多隨機的探索，可能會影響到學習的Q值。

- Q-Learning:
	- "遵守"跟"學習"是根據不同的策略。
	- 這樣可能會導致糟糕的在線學習效果。(因為遵守跟學習是不同的策略)
	- 不過在動作選擇的貪婪特性，不會影響到學習的Q值

![700](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/700.png)

- Off-Policy優點
	- Agnet在環境中所採取的動作與學習流程不再相關，有機會構建算法的不同變型體。
		- 例如，可以在採取動作時遵守更加探索性的策略，並學習最優函數。但是在某個時間點，我們可以停止探索並遵守最優策略，已獲得更好結果。
	- Agent可以通過觀察人們所示範動作的效果來學習規律(不是很懂)。
	- 離線學習或批量學習更加輕鬆，因為不用每個時間步都要更新策略。

![701](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/701.png)

<h2 id="6">深度Q網路</h2>

- [Mnih et al.，2015 年，《通过深度强化学习实现人类级别的控制》。](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- 
**Deep Q Network**

- 一次傳入一張遊戲畫面截圖，利用一個深度神經網路來當作函數的逼近器，然後生成一個動作值向量，並根據強化信號，在每個時間步將遊戲得分的變化往回反饋。

![702](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/702.png)

- **輸入端**: Atari的遊戲的分辨率210*160，每個像素有128種可能的顏色，雖然是一個離散狀態空間，都處理起來還是相當龐大。
	- 所以DeepMind團隊為了降低複雜性，將畫面縮到84*84的正方形圖片
		- 正方形圖片使他們能夠在GPU上使用更優化的神經網路運算。
	- 然後一次取4張這樣的圖片疊加。

![703](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/703.png)

- **輸出端**: 與傳統強化學習(一次僅生成一個Q值)不同的地方是，此深度網路會同時為所有可能的動作各生成一個Q值。

![704](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/704.png)

- **神經網路設計**: 
	- 遊戲畫面截圖利用CNN進行處理，使得系統能夠發現空間關係並探索空間。
	- 因為是一次傳入4張圖片，CNN可以提取這4張圖片的時間屬性。
	- 架構是兩層con layer + 兩層 full layer。

![705](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/705.png)

- 訓練這樣的網路需要大量的數據，即使有這樣的數據，也不能保證收斂到最優值函數，實際上，某些情況下，網路權重會因為動作和狀態的之間的關係非常的緊密而震盪或發散，這樣會導致產生出非常不穩定並且效率很低的策略，因此DeepMind團隊使用了很多技巧來修改Q學習算法。
	- **經驗回放**
	- **固定Q目標**

![706](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/706.png)

<h2 id="7">經驗回放</h2>

- 更有效率的使用已經觀察過的經驗。
- 基本的在線Q學習算法，獲得一個動作，狀態，獎勵，跟下個狀態，從中學習規律，然後丟棄。

![707](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/707.png)

- **Reply Buffer**: 可能之前某些學習經驗中的狀態很少遇到，或者某些動作代價很高，將這些學習過的經驗位元組儲存，進行有效的重複學習這些寶貴的經驗。
	- 另外，根據經驗回放，我們可以隨機從buffer中取樣，這樣有助於打破相互之間關係的聯繫，並最終防止動作值震盪或嚴重發散。

![708](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/708.png)

**經驗回放**

- 就是構建一個樣本數據庫，如此一來可以將強化學習(至少是在值學習的部分)簡化成監督學習。
- 將罕見或更重要的經驗的位元組調高其優先級。

![709](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/709.png)


<h2 id="8">固定Q目標</h2>

- 經驗回放，可以幫助我們解決一種類型的關聯，即位元組之前的連續經驗。
- Q學習還容易收到另外一種聯繫的影響。
	- Q學習是一種時間差分(TD)學習，我們目標是縮小TD Target跟當前預測Q值之間的差異。
	- 其中TD Target正確來說應該要替換成真正的值函數q(S,A)，但我們不知道真正的值函數是什麼。

![710](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/710.png)

- 一開始，我們使用q(S,A)定義平方損失誤差

![711](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/711.png)

- 並針對w進行偏微分，獲得梯度下降更新規則，q(S,A)並不依賴我們的函數逼近器或者其他參數，因此形成簡單的導數，即更新規則。
	- **但TD Target卻依賴這些參數，因此將真值函數q(S,A)替換成TD Target在數學上是不成立的**

![712](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/712.png)

- **問題點**: 我們的目標和我們要更改的參數之間有聯繫。
	- 也就是當我們進行參數的更新時，我們的目標也跟著改變，就像一直在追著一個會移動的目標。

![713](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/713.png)
	
**固定Q目標**

- 我們將W換W-(固定)，用這個W-生成目標和更改W，並持續一定數量的學習步驟。
- 然後再利用最新的W來更新W-，在重複進行學習。
- 這樣可以使目標跟參數拆分出來，使學習更加的穩定。

![714](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/714.png)

<h2 id="9">深度Q學習算法</h2>

- [Mnih et al.，2015 年，《通过深度强化学习实现人类级别的控制》。](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

- **SAMPLE**: 對環境取樣。
	- 執行動作。
	- 並將經驗儲存到資料庫中。

- **LEARN**: 學習經驗。 
	- 從資料庫中隨機抽取批量的經驗。
	- 使用梯度下降來學習批次經驗中的規律。

- **SAMPLE**和**LEARN**並非互相依賴，可以完成多次SAMPLE，然後進行一次LEARN，或者具有不同隨機批次的LEARN。

![715](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/715.png)

- 算法的其他內容主要就是支持**SAMPLE**和**LEARN**這兩個步驟。
	- 初始化reply memory的大小N(因為memory大小有限，所以只會保留最近N個經驗元組)
	- 初始化神經網路的參數W
	- 初始化第二組參數W-，可以設置跟W一樣。
	- 對於每個階段，還有該階段中每個時間步t
		- 會輸入原始遊戲螢幕圖像(需轉為灰階，並採剪成正方形)
		- 此外，為了捕獲時間關係，可以堆疊這些圖像以構建每個狀態向量。
		- 利用一個函數phi來表示此預處理跟堆疊操作。

![716](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/716.png)

<h2 id="10">DQN改進</h2>

- 之後，有許多論文在設法改良DQN，其中最重要的三個改進。
	- 雙DQN
	- 優先回放
	- 對抗網路

[van Hasselt et al.，2015 年，《双 Q 学习的深度强化学习》](https://arxiv.org/abs/1509.06461)

[Schaul et al.，2016 年，《优先经验回放》](https://arxiv.org/abs/1511.05952)

[Wang et al.，2015 年。《深度强化学习的对抗网络架构》](https://arxiv.org/abs/1511.06581)

![717](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/717.png)

- **第一個問題**: Q學習容易出現過高的估計值問題。
	- 重寫TD Target並展開max運算。
		- 想要獲得S'的Q值，以及該狀態中所有潛在動作中實現最大Q值得動作。
		- 其中arg max運算有可能會出錯，我們可能還沒收集到足夠的訊息來判斷最佳的動作是什麼，卻每次都挑出最大的，所以很容易出現高估的狀況。

![718](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/718.png)

**雙DQN**

- 利用兩組不同的參數W,W-進行選擇並評估。

![719](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/719.png)

- **第二個問題**: 我們隨機從reply buffer中挑選經驗，並進行訓練，但某些經驗可能比其他經驗更重要更需要學習，且因為buffer大小有限，因此一些早期的重要經驗可能會被刪除。

![720](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/720.png)

**優先回放**

- 根據TD error的大小為reply buffer中的過往經驗進行優先分級。誤差越大表示從該經驗中學到的規律越多。再將每個誤差進行標準化就成了我們的**sample probability**。
	- 常量e: 確保那些TD error=0的經驗也是有機會被挑選到。
	- a次方: 用來控制隨機性或者優先性的幅度。a=1，表示只採用優先性。
	- N: 抽樣權重，是reply buffer的大小。

![721](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/721.png)

**對抗網路**

![722](https://github.com/htaiwan/note-Udacity-machine-learning/blob/master/Assets/722.png)